import torch
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
import sys
from pathlib import Path


class GCN(torch.nn.Module):
    def __init__(self, in_channels=3, num_classes=3, dropout=0.3):
        super().__init__()

        self.conv1 = GCNConv(in_channels, 64)
        self.conv2 = GCNConv(64, 128)
        self.conv3 = GCNConv(128, 256)
        self.conv4 = GCNConv(256, 128)

        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(256)
        self.bn4 = torch.nn.BatchNorm1d(128)

        self.lin1 = torch.nn.Linear(128 * 2, 64)
        self.lin2 = torch.nn.Linear(64, num_classes)

        self.dropout = dropout

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x1 = torch.nn.functional.relu(self.bn1(self.conv1(x, edge_index)))
        x2 = torch.nn.functional.relu(self.bn2(self.conv2(x1, edge_index)))
        x3 = torch.nn.functional.relu(self.bn3(self.conv3(x2, edge_index)))
        x4 = torch.nn.functional.relu(self.bn4(self.conv4(x3, edge_index)))

        x_pool = torch.cat([global_mean_pool(x4, batch), global_max_pool(x4, batch)], dim=1)

        x_pool = torch.nn.functional.dropout(x_pool, p=self.dropout, training=self.training)
        x_pool = torch.nn.functional.relu(self.lin1(x_pool))
        x_pool = self.lin2(x_pool)

        return x_pool
