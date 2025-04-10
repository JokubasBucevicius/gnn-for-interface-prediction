'''
Script for loading and cleaning the data (creating PyG datasets)
'''


import os
import pandas as pd
import random
import torch



class DataLoader:
    def __init__(self, base_path: str, batch_size: int=5, mode: str = "binary"):
        """
        Initializes the data loader.

        :param base_path (str): Path to the main directory (e.g., 'graphs/k95/dgDNR')
        :param batch_size (int): Number of protein graphs to load at once
        :param mode (str): Classification mode - "binary" (binds to NA or not) or "multiclass" (to which NA binds) (this changes the data cleaning step)
        """
        self.base_path = base_path
        self.batch_size = batch_size
        self.mode = mode.lower()
        assert self.mode in ["binary", "multiclass"], "Mode must be 'binary' or 'multiclass'"

        # Initializint stats for each residue
        self.residue_counts = {i: 0 for i in range(20)}
        self.residue_bindings = {i: 0 for i in range(20)}
        self.binding_probabilities = {}


    def get_pdb_folders(self):
        """Returns a list of available PDB protein folders in the base path."""
        all_items = os.listdir(self.base_path)
        pdb_folders = [f for f in all_items if os.path.isdir(os.path.join(self.base_path, f))]
        pdb_folders = sorted(pdb_folders)
        print(f"Found {len(pdb_folders)} PDB folders, using {len(pdb_folders[:self.batch_size])}")
        pdb_folders = pdb_folders[:self.batch_size]
        print("Loaded PDB IDs:", ", ".join(pdb_folders))
        return pdb_folders


    def load_protein_graphs(self):
        """
        Loads nodes and edges for a small batch of proteins from the given base path.
        Returns a list of (pdb_id, nodes_df, edges_df) tuples.
        """
        pdb_folders = self.get_pdb_folders()
        protein_graphs = []

        for pdb_id in pdb_folders:
            pdb_path = os.path.join(self.base_path, pdb_id)
            nodes_path = os.path.join(pdb_path, 'graph_nodes.csv')
            edges_path = os.path.join(pdb_path, 'graph_links.csv')

            if not os.path.exists(nodes_path) or not os.path.exists(edges_path):
                print(f"Skipping {pdb_id} (missing files)")
                continue
                
            nodes_df = pd.read_csv(nodes_path)
            nodes_df = self.clean_nodes(nodes_df)

            edges_df = pd.read_csv(edges_path)
            edges_df = self.clean_edges(edges_df)

            residue_nodes_df = self.aggregate_atoms_nodes(nodes_df)
            residue_edges_df = self.aggregate_atoms_edges(edges_df)

            residue_embeddings = self.get_embeddings(pdb_id, residue_nodes_df)
            ##Optional (comment if not used)
            # edges_df = self.calculate_edge_weights(nodes_df, edges_df)
            
            pyg_graph = self.convert_to_pyg_graph(residue_nodes_df, residue_edges_df, residue_embeddings)


            protein_graphs.append({
                "pdb_id": pdb_id,
                "nodes": residue_nodes_df,
                "edges": residue_edges_df,
                "pyg_graph": pyg_graph
            })

        
        print(f"Successfully loaded {len(protein_graphs)} protein graphs (batch size = {self.batch_size})")

        # Calculate and log binding probabilities after loading
        self.finalize_binding_probabilities()

        print("Residue binding probabilities (calculated from loaded data):")
        for res_type, prob in self.binding_probabilities.items():
            print(f"Residue {res_type}: {prob:.3f}")
        print("-" * 40)


        return protein_graphs
    
    def load_and_split(self, train_ratio=0.8, shuffle=True):
        protein_graphs = self.load_protein_graphs()
        pdb_ids = [g["pdb_id"] for g in protein_graphs]

        if shuffle:
            combined = list(zip(pdb_ids, protein_graphs))
            random.shuffle(combined)
            pdb_ids, protein_graphs = zip(*combined)

        total = len(protein_graphs)
        if total == 1:
            train_graphs = protein_graphs
            test_graphs = []
        else:
            split_idx = max(1, int(total * train_ratio))
            train_graphs = protein_graphs[:split_idx]
            test_graphs = protein_graphs[split_idx:]
        
        train_pdbs = pdb_ids[:len(train_graphs)]
        test_pdbs = pdb_ids[len(train_graphs):]

        print(f"Training set ({len(train_graphs)} proteins): {', '.join(train_pdbs)}")
        print(f"Test set ({len(test_graphs)} proteins): {', '.join(test_pdbs)}")

        return train_graphs, test_graphs
    
    def aggregate_atoms_nodes(self, nodes_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregates atom-level features into residue-level features using ID_resSeq.
        """
        if "ID_resSeq" not in nodes_df.columns:
            raise ValueError("Expected 'ID_resSeq' column for residue-level grouping")
        
        if "surface_atom" not in nodes_df.columns:
            nodes_df["surface_atom"] = (nodes_df["sas_area"] > 0.0).astype(int)
        # Pooling strategy for residue-level features
        pooling = {
            "residue_type": "first",
            "sas_area": "sum",
            "voromqa_score_r": "first",
            "volume": "sum",
            "center_x": "mean",
            "center_y": "mean",
            "center_z": "mean",
            "surface_atom": "max",
        }

        if self.mode == "binary":
            pooling["bsite"] = "max"
        else:
            pooling["ssDNA_bind"] = "max"
            pooling["dsDNA_bind"] = "max"
            pooling["RNA_bind"] = "max"

        residue_nodes_df = nodes_df.groupby("ID_resSeq").agg(pooling).reset_index()

        # Residue grouping function
        def map_residue_to_group(residue_type):
            charged = {0, 7, 11, 18}    # Arg, Lys, His, Asp
            polar = {1, 3, 8, 13, 15}   # Ser, Gln, Asn, Thr, Cys
            hydrophobic = {2, 5, 6, 9, 10, 12, 14, 17}  # Ala, Ile, Leu, Met, Phe, Val, Trp, Pro
            aromatic = {4, 16}          # Tyr, His (His is both charged and aromatic)

            if residue_type in charged:
                return 0  # Charged
            elif residue_type in polar:
                return 1  # Polar
            elif residue_type in hydrophobic:
                return 2  # Hydrophobic
            elif residue_type in aromatic:
                return 3  # Aromatic
            else:
                return 4  # Unknown (should never happen)
            
        residue_nodes_df["residue_group"] = residue_nodes_df["residue_type"].map(map_residue_to_group)

        return residue_nodes_df
    
    def aggregate_atoms_edges(self, edges_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregates atom-level contact edges into residue-level edges
        using ID_resSeq1 and ID_resSeq2.
        """

        required_columns = {"ID1_resSeq", "ID2_resSeq", "area", "distance"}
        if not required_columns.issubset(edges_df.columns):
            raise ValueError(f"Missing required columns in edge dataframe: {required_columns - set(edges_df.columns)}")

        # Remove self-contacts
        edges_df = edges_df[edges_df["ID1_resSeq"] != edges_df["ID2_resSeq"]]

        # Ensure consistent edge direction (to treat undirected edges equally)
        edges_df["res1"] = edges_df[["ID1_resSeq", "ID2_resSeq"]].min(axis=1)
        edges_df["res2"] = edges_df[["ID1_resSeq", "ID2_resSeq"]].max(axis=1)

        # Pool features between each residue pair
        residue_edges_df = edges_df.groupby(["res1", "res2"]).agg({
            "area": "sum",
            "distance": "mean",
            "voromqa_energy" : "sum"
            
        }).reset_index()

        # Rename columns to match edge convention
        residue_edges_df = residue_edges_df.rename(columns={
            "res1": "ID1_resSeq",
            "res2": "ID2_resSeq"
        })

        return residue_edges_df


    def clean_nodes(self, nodes_df: pd.DataFrame) -> pd.DataFrame:
        """ (Helper function) Cleans the nodes DataFrame to retain only useful columns for GNN. """

        base_columns = [
            "atom_index", "residue_index",
            "residue_type",
            "sas_area", "voromqa_score_r",
            "volume",
            "center_x", "center_y", "center_z", "ID_resSeq"
        ]

        if self.mode == "binary":
            label_column = ["bsite"]  # Binary classification (binds or not)
        else:  # self.mode == "multiclass"
            label_column = ["ssDNA_bind", "dsDNA_bind", "RNA_bind"]  # Multi-class labels

        # Final columns to keep
        columns_to_keep = base_columns + label_column

        nodes_df = nodes_df[columns_to_keep].copy()

        # Keep only the columns that exist (avoids KeyError if some are missing)
        nodes_df = nodes_df[[col for col in columns_to_keep if col in nodes_df.columns]]

        # Surface detection
        nodes_df["surface_atom"] = (nodes_df["sas_area"] > 0.0).astype(int)

        # Residue binding stats
        for _, row in nodes_df.iterrows():
            residue_type = row["residue_type"]
            self.residue_counts[residue_type] += 1
            if row["bsite"] == 1:
                self.residue_bindings[residue_type] += 1

        # Ensure labels are properly formatted
        if self.mode == "multiclass":
            nodes_df["bind_type"] = nodes_df[["ssDNA_bind", "dsDNA_bind", "RNA_bind"]].idxmax(axis=1)
            nodes_df = nodes_df.drop(columns=["ssDNA_bind", "dsDNA_bind", "RNA_bind"])

        # Function to normalize numerical column values to [0-1]
        def normalize_column(df, col_name):
            min_val = df[col_name].min()
            max_val = df[col_name].max()
            if max_val > min_val:
                df[col_name] = (df[col_name] - min_val) / (max_val - min_val)
            else:
                df[col_name] = 0.0  # if all values are the same
            return df
        to_normalize = ["sas_area", "volume", "center_x", "center_y", "center_z"]
        for col in to_normalize:
            nodes_df = normalize_column(nodes_df, col)
        return nodes_df

    def clean_edges(self, edges_df: pd.DataFrame) -> pd.DataFrame:
        """ (Helper function) Cleans the edges DataFrame to retain only useful columns for GNN. """
        columns_to_keep = [
            "ID1_resSeq", "ID2_resSeq",
            "area", "distance", "voromqa_energy"
              
        ]

        edges_df = edges_df[[col for col in columns_to_keep if col in edges_df.columns]]

        return edges_df
    
    def get_embeddings(self, pdb_id, residue_nodes_df, embedding_dir ="/home/jokubas/Magistras/duomenys/emb/ESM2/be_klasterio/dgDNR", layer=6):
      
        

        
        emb_path = os.path.join(embedding_dir, f"{pdb_id[:6]}.pt")

        if not os.path.exists(emb_path):
            print(f"Embedding not found for {pdb_id}, skipping...")
            return None
            

        try:
            emb_data = torch.load(emb_path)
            full_embeddings = emb_data["representations"][layer]  # shape: [L, D]

                
            id_resseqs = residue_nodes_df["ID_resSeq"].values - 1 # because ID numbering starts from 1
            id_resseqs_tensor = torch.tensor(id_resseqs, dtype=torch.long)

            filtered_embeddings = full_embeddings[id_resseqs_tensor]
            return filtered_embeddings  

        except Exception as e:
            print(f"Failed to load/filter embeddings for {pdb_id}: {e}")
            return None



    
    def calculate_edge_weights(self, nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates edge weights based on solvent-accessible surface area (sas_area)
        of the two atoms connected by each edge. Adds a column 'sas_area_weight'.
        """

        # Build a lookup table: atom_index -> sas_area
        sas_area_lookup = nodes_df.set_index("ID_resSeq")["sas_area"].to_dict()

        # Compute sas_area_weight for each edge
        edge_sas_weights = [
        (sas_area_lookup[row["ID1_resSeq"]] + sas_area_lookup[row["ID2_resSeq"]]) / 2
        for _, row in edges_df.iterrows()
        ]

        # Add the new column to edges_df
        edges_df = edges_df.copy()
        edges_df["sas_area_weight"] = edge_sas_weights

        return edges_df

    def calculate_binding_probabilities(self, protein_graphs):
        '''
        Calculating binding probabilities for each amino acid this number will be used later to adjusting the weights for each amino acid
        '''
        residue_counts = {i: 0 for i in range(20)}  # 20 residue types
        residue_bindings = {i: 0 for i in range(20)}

        for graph in protein_graphs:
            nodes = graph["nodes"]
            for _, row in nodes.iterrows():
                residue_counts[row.residue_type] += 1
                if row.bsite == 1:
                    residue_bindings[row.residue_type] += 1

        binding_probabilities = {}
        for residue_type, count in residue_counts.items():
            if count > 0:
                binding_probabilities[residue_type] = residue_bindings[residue_type] / count
            else:
                binding_probabilities[residue_type] = 0.0

        return binding_probabilities

    def finalize_binding_probabilities(self):
        for res_type, count in self.residue_counts.items():
            if count > 0:
                self.binding_probabilities[res_type] = self.residue_bindings[res_type] / count
            else:
                self.binding_probabilities[res_type] = 0.0

    def convert_to_pyg_graph(self, residue_nodes_df, residue_edges_df, residue_embeddings):
        """
        Converts cleaned nodes and edges dataframes into a PyTorch Geometric Data object.
        """
        import torch
        from torch_geometric.data import Data  # type: ignore

        continuous_columns = [
            "sas_area", "voromqa_score_r",
            "volume",
            "center_x", "center_y", "center_z",
            "surface_atom"
        ]

        cont_feats = torch.tensor(residue_nodes_df[continuous_columns].values, dtype=torch.float)
        
        residue_type=torch.tensor(residue_nodes_df["residue_type"].values, dtype=torch.float).unsqueeze(1)
        residue_group=torch.tensor(residue_nodes_df["residue_group"].values, dtype=torch.float).unsqueeze(1)
        
        cont_feats = torch.cat([cont_feats, residue_type, residue_group, residue_embeddings], dim=1) 
        # Node labels (y)
        if self.mode == "binary":
            y = torch.tensor(residue_nodes_df["bsite"].values, dtype=torch.long)
        else:  # multiclass
            y = torch.tensor(residue_nodes_df["bind_type"].map(bind_type_mapping).values, dtype=torch.long)  # type: ignore

        # Edge index (nodes connectivity)
        # Re-index residues
        residue_ids = residue_nodes_df["ID_resSeq"].tolist()
        id_map = {rid: idx for idx, rid in enumerate(residue_ids)}

        edge_index = residue_edges_df[["ID1_resSeq", "ID2_resSeq"]].applymap(id_map.get).values.T
        edge_index = torch.tensor(edge_index, dtype=torch.long)

        # Edge attributes (area and distance)
        edge_attr = torch.tensor(residue_edges_df[["area", "distance"]].values, dtype=torch.float)

        num_nodes = residue_nodes_df.shape[0]
        # Create PyTorch Geometric Data object
        graph_data = Data(
            cont_feats = cont_feats,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            num_nodes=num_nodes
            )
        return graph_data



