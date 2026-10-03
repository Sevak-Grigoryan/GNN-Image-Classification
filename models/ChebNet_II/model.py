import torch
import torch_geometric
from Cheb_Layer import ChebNetIIConv

class ChebNetII(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = ChebNetIIConv(in_channels = 3, out_channels= 16, K = 2)
        print("Alpha:", self.conv1.alpha)  
        self.bn1 = torch_geometric.nn.BatchNorm(in_channels=16)
        self.conv2 = ChebNetIIConv(in_channels = 16, out_channels = 32, K = 2)
        print("Alpha2:", self.conv1.alpha)  

        self.bn2 = torch_geometric.nn.BatchNorm(in_channels=32)
        self.conv3 = ChebNetIIConv(in_channels = 32, out_channels = 64, K = 2)
        print("Alpha3:", self.conv1.alpha)  

        self.bn3 = torch_geometric.nn.BatchNorm(in_channels=64)

        self.l1 = torch.nn.Linear(in_features=64, out_features=32)
        self.l2 = torch.nn.Linear(in_features=32, out_features=3)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = torch.nn.functional.relu(self.bn1(self.conv1(x, edge_index)))
        x = torch.nn.functional.dropout(x, p=0.5, training=self.training)
        x = torch.nn.functional.relu(self.bn2(self.conv2(x, edge_index)))
        x = torch.nn.functional.dropout(x, p=0.5, training=self.training)
        x = torch.nn.functional.relu(self.bn3(self.conv3(x, edge_index)))
        x = torch.nn.functional.dropout(x, p=0.5, training=self.training)

        x = torch_geometric.nn.global_mean_pool(x, batch)

        x = torch.nn.functional.relu(self.l1(x))
        x = torch.nn.functional.relu(self.l2(x))

        return x
