import torch
import torch_geometric
# class GNN(torch.nn.Module):

#     def __init__(self):
#         super().__init__()

#         self.gat1 = torch_geometric.nn.GATConv(
#             in_channels=32, out_channels=32, heads=2, concat=True
#             )
#         self.norm1 = torch_geometric.nn.GraphNorm(64)
#         self.gat2 = torch_geometric.nn.GATConv(
#             in_channels=64, out_channels=32, heads=2, concat=True
#             )
#         self.norm2 = torch_geometric.nn.GraphNorm(64)
#         self.lin1 = torch.nn.Linear(64, 32)
#         self.lin2 = torch.nn.Linear(32, 16)
#         self.lin3= torch.nn.Linear(16, 3)


#     def forward(self, data):
#         x, edge_index, batch = data.x, data.edge_index, data.batch

#         x = torch.nn.functional.elu(self.gat1(x, edge_index))
#         x = torch.nn.functional.elu(self.gat2(x, edge_index))

#         x = torch_geometric.nn.global_mean_pool(x, batch)
#         x = torch.nn.functional.elu(self.lin1(x))
#         x = torch.nn.functional.elu(self.lin2(x))

#         return torch.nn.functional.elu(self.lin3(x))
    


import torch
import torch.nn.functional as F
from torch_geometric.nn import GINConv, GraphNorm, global_mean_pool

class MLP(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(channels, channels),
            torch.nn.ReLU(),
            torch.nn.Linear(channels, channels),
        )

    def forward(self, x):
        return self.net(x)

class GNN(torch.nn.Module):
    def __init__(self, hidden=32, num_classes=3):
        super().__init__()

        # Encoder: map input 3 channels -> hidden
        self.encoder = torch.nn.Linear(32, hidden)

        # 3 GINConv layers
        self.conv1 = GINConv(MLP(hidden), train_eps=True)
        self.conv2 = GINConv(MLP(hidden), train_eps=True)
        self.conv3 = GINConv(MLP(hidden), train_eps=True)

        # GraphNorm layers
        self.norm1 = GraphNorm(hidden)
        self.norm2 = GraphNorm(hidden)
        self.norm3 = GraphNorm(hidden)

        # Classifier
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden, hidden // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(hidden // 2, num_classes)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.encoder(x)

        x = self.conv1(x, edge_index)
        x = self.norm1(x, batch)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = self.norm2(x, batch)
        x = F.relu(x)

        x = self.conv3(x, edge_index)
        x = self.norm3(x, batch)
        x = F.relu(x)

        x = global_mean_pool(x, batch)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    import os, sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # project root
    from utils.count_parameters import count_parameters
    model = GNN(hidden=32)
    print("GNN Parametors:")
    count_parameters(model=GNN())
