import os
import sys
import torch

# Add the parent directory (where your data_loader.py lives) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_loader import DataLoader  # import your DataLoader class

# Initialize DataLoader
loader = DataLoader(
    base_path="/home/jokubas/Magistras/duomenys/grafai/graphs/k95/dgDNR",  # adjust path as needed
    batch_size=3,
    mode="binary"
)

# Load protein graphs
protein_graphs = loader.load_protein_graphs()

# Loop through each loaded graph
for i, protein in enumerate(protein_graphs):
    print(f"\n🧬 Protein {i+1}: {protein['pdb_id']}")
    pyg_graph = protein["pyg_graph"]

    print(f"📊 Node features shape:      {pyg_graph.cont_feats.shape}")  # [N, F]
    print(f"🔗 Edge index shape:         {pyg_graph.edge_index.shape}")  # [2, E]
    print(f"📐 Edge attributes shape:    {pyg_graph.edge_attr.shape}")   # [E, D]
    print(f"🏷️  Labels shape:            {pyg_graph.y.shape}")           # [N]

    # Show a few node features and edge indices
    print("\n🔍 First 3 node features:")
    print(pyg_graph.cont_feats[:3])

    print("\n🔗 First 3 edges (index):")
    print(pyg_graph.edge_index[:, :3])

    print("\n🧩 First 3 edge attributes:")
    print(pyg_graph.edge_attr[:3])

    print("-" * 50)

