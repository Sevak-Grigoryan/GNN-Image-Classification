import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from torch_geometric.nn.norm import GraphNorm


class GAT(torch.nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        self.norm0 = GraphNorm(3)

        # 3 -> 64
        self.gat1 = GATConv(3, 32, heads=2, concat=True)   # 64
        self.norm1 = GraphNorm(64)

        # 64 -> 96
        self.gat2 = GATConv(64, 16, heads=6, concat=True)  # 96
        self.norm2 = GraphNorm(96)

        # residual: 64 -> 96
        self.res_linear = torch.nn.Linear(64, 96)

        # 96 -> 32  (heads=1, որ pooling-ից հետո լինի 64)
        self.gat3 = GATConv(96, 32, heads=1, concat=True)  # 32
        self.norm3 = GraphNorm(32)

        self.fc1 = torch.nn.Linear(64, 32)   # mean+max = 32+32
        self.fc2 = torch.nn.Linear(32, num_classes)

        self.dropout = 0.4

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.norm0(x, batch)

        x1 = self.gat1(x, edge_index)      # [N, 64]
        x1 = self.norm1(x1, batch)
        x1 = F.elu(x1)
        x1 = F.dropout(x1, p=self.dropout, training=self.training)

        x2 = self.gat2(x1, edge_index)     # [N, 96]
        x2 = self.norm2(x2, batch)
        x2 = x2 + self.res_linear(x1)      # [N, 96]
        x2 = F.elu(x2)
        x2 = F.dropout(x2, p=self.dropout, training=self.training)

        x = self.gat3(x2, edge_index)      # [N, 32]
        x = self.norm3(x, batch)
        x = F.elu(x)

        x = torch.cat(
            [
                global_mean_pool(x, batch), 
                global_max_pool(x, batch)
             ],
            dim=1
        )                                   # [B, 64]

        x = F.relu(self.fc1(x))            # [B, 32]
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)

        return x