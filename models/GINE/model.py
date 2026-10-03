import torch
import torch.nn.functional as F
from torch_geometric.nn.norm import GraphNorm
from torch_geometric.nn.conv import GINEConv
from torch_geometric.nn.pool import global_max_pool

class GINE(torch.nn.Module):
    def __init__(self, num_classes=3, node_in_dim=3, hidden_dim=48, edge_dim=5, dropout=0.2):
        super().__init__()
        self.dropout = dropout

        # Node encoder
        self.node_encoder = torch.nn.Linear(node_in_dim, hidden_dim)
        
        # Edge encoder - projects edge features to hidden_dim
        self.edge_encoder = torch.nn.Linear(edge_dim, hidden_dim)

        # GINE layers with GraphNorm
        self.conv1 = GINEConv(
            torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim)
            ),
            edge_dim=hidden_dim,  # Changed to hidden_dim
            train_eps=True
        )
        self.norm1 = GraphNorm(hidden_dim)
        
        self.conv2 = GINEConv(
            torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim)
            ),
            edge_dim=hidden_dim,  # Changed to hidden_dim
            train_eps=True
        )
        self.norm2 = GraphNorm(hidden_dim)
        
        self.conv3 = GINEConv(
            torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim)
            ),
            edge_dim=hidden_dim,  # Changed to hidden_dim
            train_eps=True
        )
        self.norm3 = GraphNorm(hidden_dim)

        # Classifier
        self.lin1 = torch.nn.Linear(hidden_dim, hidden_dim // 2)
        self.lin2 = torch.nn.Linear(hidden_dim // 2, num_classes)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # Encode nodes and edges
        x = F.relu(self.node_encoder(x))
        edge_attr = F.relu(self.edge_encoder(edge_attr))  # Encode edge features

        # GINE layer 1
        h = self.conv1(x, edge_index, edge_attr)
        h = self.norm1(h, batch)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        x = x + h

        # GINE layer 2
        h = self.conv2(x, edge_index, edge_attr)
        h = self.norm2(h, batch)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        x = x + h

        # GINE layer 3
        h = self.conv3(x, edge_index, edge_attr)
        h = self.norm3(h, batch)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        x = x + h

        # Global pooling
        x = global_max_pool(x, batch)

        # Classifier
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        return x

if __name__ == "__main__":
    import os, sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # project root
    from utils.count_parameters import count_parameters
    count_parameters(model=GINE())
