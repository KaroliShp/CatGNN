"""
GCN architecture from:
https://github.com/pyg-team/pytorch_geometric/blob/master/benchmark/kernel/gcn.py
"""

import torch
from torch.nn import Linear
import torch_geometric.nn

from catgnn.layers.gcn.gcn_mpnn_2 import GCNLayer_MPNN_2


class GCN_2(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden):
        super().__init__()
        self.conv1 = GCNLayer_MPNN_2(dataset.num_features, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GCNLayer_MPNN_2(hidden, hidden))
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
        # Create vertices - can probably do this elsewhere, would be faster
        V = torch.arange(0, X.shape[0], dtype=torch.int64, device=E.device)
        X = torch.nn.functional.relu(self.conv1(V, E, X))
        for conv in self.convs:
            X = torch.nn.functional.relu(conv(V, E, X))
        X = torch_geometric.nn.global_mean_pool(X, batch)
        X = torch.nn.functional.relu(self.lin1(X))
        X = torch.nn.functional.dropout(X, p=0.5, training=self.training)
        return self.lin2(X)  # Used to be log softmax

    def __repr__(self):
        return self.__class__.__name__


class PyG_GCN(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden):
        super().__init__()
        self.conv1 = torch_geometric.nn.GCNConv(dataset.num_features, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(torch_geometric.nn.GCNConv(hidden, hidden))
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
        x = torch.nn.functional.relu(self.conv1(x, edge_index))
        for conv in self.convs:
            x = torch.nn.functional.relu(conv(x, edge_index))
        x = torch_geometric.nn.global_mean_pool(x, batch)
        x = torch.nn.functional.relu(self.lin1(x))
        x = torch.nn.functional.dropout(x, p=0.5, training=self.training)
        return self.lin2(x)  # Used to be log softmax

    def __repr__(self):
        return self.__class__.__name__
