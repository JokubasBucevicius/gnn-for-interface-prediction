import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

class ProteinGAT(torch.nn.Module):
    def __init__(self, hidden_dim, output_dim, heads=4, mode="binary"):
        super(ProteinGAT, self).__init__()

        self.mode = mode.lower()  # "binary" or "multiclass"

        edge_dim = 2

        # Continues features from DataLoader
        self.num_cont_feats = 329 # sas_area, voromqa scores, volume, radius, x/y/z, surface_atom, 320 embeddings from ESM2
        
        input_dim = (
            self.num_cont_feats
        ) 
        
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, edge_dim=edge_dim)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, edge_dim=edge_dim)

        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, data):

        # Concat embeddings + continous features
        x =  data.cont_feats
        x = self.conv1(x, data.edge_index, data.edge_attr)
        x = F.elu(x)
        x = self.conv2(x, data.edge_index, data.edge_attr)

        out = self.fc(x)  # shape: [n_nodes, output_dim]

        # if self.mode == "binary":
        #     return torch.sigmoid(out).squeeze(-1)  # shape: [n_nodes]
        # else:
        #     return F.log_softmax(out, dim=-1)  # shape: [n_nodes, num_classes]
        return out.squeeze(-1)
    