import torch
from torch.nn import Linear
import torch_geometric.nn

from catgnn.layers.gcn.gcn_mpnn_2 import GCNLayer_MPNN_2


"""
GCN
https://github.com/pyg-team/pytorch_geometric/blob/master/benchmark/kernel/gcn.py
"""


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
        x, edge_index, batch = data.x, data.edge_index, data.batch
        v = torch.arange(0, x.shape[0], dtype=torch.int64)
        x = torch.nn.functional.relu(self.conv1(v, edge_index, x))
        for conv in self.convs:
            x = torch.nn.functional.relu(conv(v, edge_index, x))
        x = torch_geometric.nn.global_mean_pool(x, batch)
        x = torch.nn.functional.relu(self.lin1(x))
        x = torch.nn.functional.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return torch.nn.functional.log_softmax(x, dim=-1)

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
        x = self.lin2(x)
        return torch.nn.functional.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__