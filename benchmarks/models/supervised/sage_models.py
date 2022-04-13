"""
GraphSAGE architecture from:
https://github.com/pyg-team/pytorch_geometric/blob/master/benchmark/kernel/graph_sage.py
"""

import torch
import torch.nn.functional as F
from torch.nn import Linear
import torch_geometric
from catgnn.layers.sage.sage_mpnn_2 import SAGELayer_MPNN_2


class GraphSAGE_2(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden):
        super().__init__()
        self.conv1 = SAGELayer_MPNN_2(dataset.num_features, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(SAGELayer_MPNN_2(hidden, hidden))
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, dataset.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        X, E, batch = data.x, data.edge_index, data.batch
        V = torch.arange(0, X.shape[0], dtype=torch.int64, device=E.device)  # Create vertices
        X = F.relu(self.conv1(V, E, X))
        for conv in self.convs:
            X = F.relu(
                conv(
                    V,
                    E,
                    X,
                )
            )
        X = torch_geometric.nn.global_mean_pool(X, batch)
        X = F.relu(self.lin1(X))
        X = F.dropout(X, p=0.5, training=self.training)
        X = self.lin2(X)
        return X  # Log softmax

    def __repr__(self):
        return self.__class__.__name__


class PyG_GraphSAGE(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden):
        super().__init__()
        self.conv1 = torch_geometric.nn.SAGEConv(dataset.num_features, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(torch_geometric.nn.SAGEConv(hidden, hidden))
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, dataset.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = torch_geometric.nn.global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__
