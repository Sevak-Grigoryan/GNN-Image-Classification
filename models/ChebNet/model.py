import torch
import torch.nn.functional as F
from torch_geometric.nn import (
    ChebConv,
    GraphNorm,
    global_mean_pool,
    global_max_pool,
    Linear
)
class ChebNet(torch.nn.Module):
    def __init__(
        self,
        in_channels=3,
        num_classes=3,
    ):
        super().__init__()

        self.conv1 = ChebConv(in_channels, 32, K=2)
        self.norm1 = GraphNorm(32)

        self.conv2 = ChebConv(32, 32, K=2)
        self.norm2 = GraphNorm(32)

        self.conv3 = ChebConv(32, 64, K=3)
        self.norm3 = GraphNorm(64)

        self.dropout = torch.nn.Dropout(0.3)

        self.fc1 = Linear(64 * 2, 64)
        self.fc2 = Linear(64, 32)
        self.fc3 = Linear(32, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Block 1
        x1 = F.relu(self.norm1(self.conv1(x, edge_index)))
        x1 = self.dropout(x1)

        # Block 2 (residual)
        x2 = F.relu(self.norm2(self.conv2(x1, edge_index)))
        x = x1 + x2
        x = self.dropout(x)

        # Block 3
        x = F.relu(self.norm3(self.conv3(x, edge_index)))

        # Pooling
        x = torch.cat([
            global_mean_pool(x, batch),
            global_max_pool(x, batch)
        ], dim=1)

        # Classifier
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

if __name__ == "__main__":
    import os, sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # project root
    from utils.count_parameters import count_parameters
    count_parameters(model=ChebNet())
