import torch
import torch.nn.functional as F
import torch_geometric
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


class GIN(torch.nn.Module):
    def __init__(self, hidden=64, num_classes=3):
        super().__init__()

        self.encoder = torch.nn.Linear(3, hidden)

        self.conv1 = GINConv(MLP(hidden), train_eps=True)
        self.conv2 = GINConv(MLP(hidden), train_eps=True)
        self.conv3 = GINConv(MLP(hidden), train_eps=True)

        self.norm1 = GraphNorm(hidden)
        self.norm2 = GraphNorm(hidden)
        self.norm3 = GraphNorm(hidden)

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
    count_parameters(model = GIN())
