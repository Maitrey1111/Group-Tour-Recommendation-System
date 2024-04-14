# data computations
import torch

# deep learning
import torch.nn as nn
from torch_geometric.nn import GCNConv

# Local import
from data import characteristics, destinations


class GCNModel(nn.Module):
    def __init__(self, layers, dropout=0.0):
        super(GCNModel, self).__init__()

        self.dropout = dropout

        self.gcn_layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.gcn_layers.append(GCNConv(layers[i], layers[i + 1]))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_weight = torch.ones(edge_index.size(1))

        for layer in self.gcn_layers:
            x = layer(x, edge_index, edge_weight=edge_weight)
        x = torch.relu(x)

        return x


def modelParams():
    trial = {
        "hidden_dim1": 400,
        "hidden_dim2": 350,
        "hidden_dim3": 300,
        "hidden_dim4": 250,
        "hidden_dim5": 200,
        "hidden_dim6": 150,
    }
    layers = [
        len(destinations),
        trial["hidden_dim1"],
        trial["hidden_dim2"],
        trial["hidden_dim3"],
        trial["hidden_dim4"],
        trial["hidden_dim5"],
        trial["hidden_dim6"],
        len(characteristics),
    ]

    return {"layers": layers}
