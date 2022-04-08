"""
GIN architecture from:
https://github.com/pyg-team/pytorch_geometric/blob/master/benchmark/kernel/gin.py
"""

import torch
from torch.nn import BatchNorm1d as BN
from torch.nn import Linear, ReLU, Sequential
import torch_geometric.nn

from catgnn.layers.gin.gin_mpnn_2 import GINLayer_MPNN_2


class GIN_2(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden):
        super().__init__()
        self.conv1 = GINLayer_MPNN_2(
            Sequential(
                Linear(dataset.num_features, hidden),
                ReLU(),
                Linear(hidden, hidden),
                ReLU(),
                BN(hidden),
            ), train_eps=True)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                GINLayer_MPNN_2(
                    Sequential(
                        Linear(hidden, hidden),
                        ReLU(),
                        Linear(hidden, hidden),
                        ReLU(),
                        BN(hidden),
                    ), train_eps=True))
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
        V = torch.arange(0, X.shape[0], dtype=torch.int64) # Create vertices
        X = self.conv1(V, E, X)
        for conv in self.convs:
            X = conv(V, E, X)
        X = torch_geometric.nn.global_mean_pool(X, batch)
        X = torch.nn.functional.relu(self.lin1(X))
        X = torch.nn.functional.dropout(X, p=0.5, training=self.training)
        return self.lin2(X)  # Used to be log softmax

    def __repr__(self):
        return self.__class__.__name__


class PyG_GIN(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden):
        super().__init__()
        self.conv1 = torch_geometric.nn.GINConv(
            Sequential(
                Linear(dataset.num_features, hidden),
                ReLU(),
                Linear(hidden, hidden),
                ReLU(),
                BN(hidden),
            ), train_eps=True)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                torch_geometric.nn.GINConv(
                    Sequential(
                        Linear(hidden, hidden),
                        ReLU(),
                        Linear(hidden, hidden),
                        ReLU(),
                        BN(hidden),
                    ), train_eps=True))
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
        x = self.conv1(x, edge_index)
        for conv in self.convs:
            x = conv(x, edge_index)
        x = torch_geometric.nn.global_mean_pool(x, batch)
        x = torch.nn.functional.relu(self.lin1(x))
        x = torch.nn.functional.dropout(x, p=0.5, training=self.training)
        return self.lin2(x)  # Used to be log softmax

    def __repr__(self):
        return self.__class__.__name__


"""
GIN0 architecture from:
https://github.com/pyg-team/pytorch_geometric/blob/master/benchmark/kernel/gin.py
"""


class GIN0_2(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden):
        super().__init__()
        self.conv1 = GINLayer_MPNN_2(
            Sequential(
                Linear(dataset.num_features, hidden),
                ReLU(),
                Linear(hidden, hidden),
                ReLU(),
                BN(hidden),
            ), train_eps=False)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                GINLayer_MPNN_2(
                    Sequential(
                        Linear(hidden, hidden),
                        ReLU(),
                        Linear(hidden, hidden),
                        ReLU(),
                        BN(hidden),
                    ), train_eps=False))
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
        V = torch.arange(0, X.shape[0], dtype=torch.int64) # Create vertices
        X = self.conv1(V, E, X)
        for conv in self.convs:
            X = conv(V, E, X)
        X = torch_geometric.nn.global_mean_pool(X, batch)
        X = torch.nn.functional.relu(self.lin1(X))
        X = torch.nn.functional.dropout(X, p=0.5, training=self.training)
        return self.lin2(X)  # Used to be log softmax

    def __repr__(self):
        return self.__class__.__name__


class PyG_GIN0(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden):
        super().__init__()
        self.conv1 = torch_geometric.nn.GINConv(
            Sequential(
                Linear(dataset.num_features, hidden),
                ReLU(),
                Linear(hidden, hidden),
                ReLU(),
                BN(hidden),
            ), train_eps=False)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                torch_geometric.nn.GINConv(
                    Sequential(
                        Linear(hidden, hidden),
                        ReLU(),
                        Linear(hidden, hidden),
                        ReLU(),
                        BN(hidden),
                    ), train_eps=False))
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
        x = self.conv1(x, edge_index)
        for conv in self.convs:
            x = conv(x, edge_index)
        x = torch_geometric.nn.global_mean_pool(x, batch)
        x = torch.nn.functional.relu(self.lin1(x))
        x = torch.nn.functional.dropout(x, p=0.5, training=self.training)
        return self.lin2(x)  # Used to be log softmax

    def __repr__(self):
        return self.__class__.__name__