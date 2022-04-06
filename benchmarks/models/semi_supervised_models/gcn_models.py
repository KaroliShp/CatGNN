from benchmarks.utils.train_semi_supervised import train_eval_loop
from catgnn.layers.gcn.gcn_mpnn_1 import GCNLayer_MPNN_1, GCNLayer_Factored_MPNN_1
from catgnn.layers.gcn.gcn_mpnn_2 import GCNLayer_MPNN_2, GCNLayer_Factored_MPNN_2, GCNLayer_MPNN_2_Forwards
import torch
from torch import nn
import torch_geometric


class GCN_1(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(GCN_1, self).__init__()
        self.gcn_layer = GCNLayer_MPNN_1(input_dim, output_dim)

    def forward(self, V, E, X):
        return self.gcn_layer(V, E, X)


class Factored_GCN_1(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(Factored_GCN_1, self).__init__()
        self.gcn_layer = GCNLayer_Factored_MPNN_1(input_dim, output_dim)

    def forward(self, V, E, X):
        return self.gcn_layer(V, E, X)


class GCN_2(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(GCN_2, self).__init__()
        self.gcn_layer = GCNLayer_MPNN_2(input_dim, output_dim)

    def forward(self, V, E, X):
        return self.gcn_layer(V, E, X)


class Factored_GCN_2(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(Factored_GCN_2, self).__init__()
        self.gcn_layer = GCNLayer_Factored_MPNN_2(input_dim, output_dim)

    def forward(self, V, E, X):
        return self.gcn_layer(V, E, X)


class GCN_2_Forwards(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(GCN_2_Forwards, self).__init__()
        self.gcn_layer = GCNLayer_MPNN_2_Forwards(input_dim, output_dim)

    def forward(self, V, E, X):
        return self.gcn_layer(V, E, X)


class PyG_GCN(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(PyG_GCN, self).__init__()
        self.gcn_layer = torch_geometric.nn.conv.GCNConv(input_dim, output_dim, cached=False, add_self_loops=True)

    def forward(self, V, E, X):
        return self.gcn_layer(X, E)

