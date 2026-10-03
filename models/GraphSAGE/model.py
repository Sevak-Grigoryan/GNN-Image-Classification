# import torch
# import torch_geometric

# class GraphSAGE(torch.nn.Module):
    
#     def __init__(self):
#         super().__init__()

#         self.conv1 = torch_geometric.nn.SAGEConv(in_channels=3, out_channels=32, aggr='mean')
#         self.b1 = torch_geometric.nn.BatchNorm(in_channels=32)
#         self.conv2 = torch_geometric.nn.SAGEConv(in_channels=32, out_channels=64, aggr='mean')
#         self.b2 = torch_geometric.nn.BatchNorm(in_channels=64)
#         self.conv3 = torch_geometric.nn.SAGEConv(in_channels=64, out_channels=128, aggr='mean')
#         self.b3 = torch_geometric.nn.BatchNorm(in_channels=128)
#         self.conv4 = torch_geometric.nn.SAGEConv(in_channels=128, out_channels=256, aggr='mean')
#         self.b4 = torch_geometric.nn.BatchNorm(in_channels=256)
#         # self.conv5 = torch_geometric.nn.SAGEConv(in_channels=256, out_channels=512, aggr='mean')
#         # self.b5 = torch_geometric.nn.BatchNorm(in_channels=512)
#         # self.l1 = torch.nn.Linear(in_features=512, out_features=256)
#         self.l1 = torch.nn.Linear(in_features=256, out_features=128)
#         self.l2 = torch.nn.Linear(in_features=128, out_features=64)
#         self.l3 = torch.nn.Linear(in_features=64, out_features=3)

#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index

#         x = self.conv1(x, edge_index)
#         x = self.b1(x)
#         x = torch.nn.functional.relu(x)
#         x = torch.nn.functional.dropout(x, p=0.5)

#         x = self.conv2(x, edge_index)
#         x = self.b2(x)
#         x = torch.nn.functional.relu(x)
#         x = torch.nn.functional.dropout(x, p=0.5)

#         x = self.conv3(x, edge_index)
#         x = self.b3(x)
#         x = torch.nn.functional.relu(x)
#         x = torch.nn.functional.dropout(x, p=0.5)

#         x = self.conv4(x, edge_index)
#         x = self.b4(x)
#         x = torch.nn.functional.relu(x)
#         x = torch.nn.functional.dropout(x, p=0.5)
       
#         # x = self.conv5(x, edge_index)
#         # x = self.b5(x)
#         # x = torch.nn.functional.relu(x)
#         # x = torch.nn.functional.dropout(x, p=0.5)


#         x = torch_geometric.nn.global_mean_pool(x, data.batch)

#         x = self.l1(x)
#         x = torch.nn.functional.relu(x)
#         x = self.l2(x)
#         x = torch.nn.functional.relu(x)
#         x = self.l3(x)
#         # x = torch.nn.functional.relu(x)
#         # x = self.l4(x)

#         return x
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool, global_max_pool, GraphNorm
from torch.nn import LayerNorm, Linear

class GraphSAGE(torch.nn.Module):
    def __init__(self):
        super().__init__()

        hidden = 70

        self.conv1 = SAGEConv(3, hidden, aggr='mean')
        self.conv2 = SAGEConv(hidden, hidden, aggr='mean')
        self.conv3 = SAGEConv(hidden, hidden, aggr='mean')
        self.conv4 = SAGEConv(hidden, hidden, aggr='mean')

        self.norm1 = GraphNorm(hidden)
        self.norm2 = GraphNorm(hidden)
        self.norm3 = GraphNorm(hidden)
        # small MLP head
        self.fc1 = Linear(hidden*2, 64)  # mean + max pooling
        self.fc2 = Linear(64, 3)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Block 1
        h = self.conv1(x, edge_index)
        x = F.relu(self.norm1(h))
        x = F.dropout(x, p=0.2, training=self.training)

        # Block 2 (residual)
        h = self.conv2(x, edge_index)
        x = x + F.elu(self.norm2(h))
        x = F.dropout(x, p=0.2, training=self.training)

        # Block 3 (residual)
        h = self.conv3(x, edge_index)
        x = x + F.elu(self.norm3(h))

        h = self.conv4(x, edge_index)
        x = x + F.elu(self.norm3(h))
        # Strong pooling
        x = torch.cat([
            global_mean_pool(x, batch),
            global_max_pool(x, batch)
        ], dim=1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

if __name__ == "__main__":
    import os, sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # project root
    from utils.count_parameters import count_parameters
    model = GraphSAGE()
    print(count_parameters(model=model))
