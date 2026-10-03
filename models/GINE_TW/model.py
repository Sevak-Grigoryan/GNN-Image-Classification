import torch
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, GraphNorm, global_max_pool

class GINE(torch.nn.Module):
    def __init__(self, num_classes=3, hidden_dim=24, edge_dim=5):
        super().__init__()

        self.edge_weight = torch.nn.Parameter(torch.empty((edge_dim, edge_dim)))
        torch.nn.init.xavier_uniform_(self.edge_weight)

        self.node_encoder = torch.nn.Linear(3, hidden_dim)
        self.edge_encoder = torch.nn.Linear(edge_dim, hidden_dim)

        def make_nn():
            return torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim//2),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim//2, hidden_dim),
            )

        self.conv1 = GINEConv(make_nn(), train_eps=True)
        self.conv2 = GINEConv(make_nn(), train_eps=True)
        self.conv3 = GINEConv(make_nn(), train_eps=True)
        self.conv4 = GINEConv(make_nn(), train_eps=True)

        self.norm1 = GraphNorm(hidden_dim)
        self.norm2 = GraphNorm(hidden_dim)
        self.norm3 = GraphNorm(hidden_dim)
        self.norm4 = GraphNorm(hidden_dim)

        self.lin1 = torch.nn.Linear(hidden_dim, hidden_dim // 2)
        self.lin2 = torch.nn.Linear(hidden_dim // 2, num_classes)

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        # safety for visualization
        if edge_attr is None:
            edge_attr = torch.ones(edge_index.size(1),1, device=x.device)

        edge_attr = edge_attr @ self.edge_weight

        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        h = F.relu(self.norm1(self.conv1(x, edge_index, edge_attr), batch))
        x = x + h

        h = F.relu(self.norm2(self.conv2(x, edge_index, edge_attr), batch))
        x = x + h

        h = F.relu(self.norm3(self.conv3(x, edge_index, edge_attr), batch))
        x = x + h

        h = F.relu(self.norm4(self.conv4(x, edge_index, edge_attr), batch))
        x = x + h

        x = global_max_pool(x, batch)
        x = F.relu(self.lin1(x))
        return self.lin2(x)

if __name__ == "__main__":
    import os, sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # project root
    from utils.count_parameters import count_parameters
    count_parameters(model=GINE())
