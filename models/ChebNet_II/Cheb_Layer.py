import torch
import torch_geometric


class ChebNetIIConv(torch.nn.Module):

    def __init__(
            self,
            in_channels = None,
            out_channels = None,
            K = None,
            init_alpha=0.5
    ):
        super().__init__()

        self.cheb_in = torch_geometric.nn.ChebConv(in_channels, out_channels, K)
        self.cheb_out = torch_geometric.nn.ChebConv(in_channels, out_channels, K)

        self.alpha = torch.nn.Parameter(torch.tensor(init_alpha))

    def forward(self, x, edge_index):

        incoming_data = edge_index.flip(0)

        outgoing_data = edge_index

        in_data = self.cheb_in(x, incoming_data)
        out_data = self.cheb_out(x, outgoing_data)

        out = self.alpha * in_data + (1 - self.alpha) * out_data
        
        return out