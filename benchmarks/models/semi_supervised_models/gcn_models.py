from benchmarks.utils.train_semi_supervised import train_eval_loop
from catgnn.layers.gcn.gcn_mpnn_1 import GCNLayer_MPNN_1, GCNLayer_Factored_MPNN_1
from catgnn.layers.gcn.gcn_mpnn_2 import GCNLayer_MPNN_2, GCNLayer_Factored_MPNN_2, GCNLayer_MPNN_2_Forwards
import torch
from torch import nn
import torch_geometric


"""
General layers
"""


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


"""
Paper benchmarking
"""

class GCN_1_Paper(torch.nn.Module):
    def __init__(self, input_dim, output_dim, factored=False):
        super().__init__()
        if factored:
            print('Factored implementation')
            self.conv1 = GCNLayer_Factored_MPNN_1(input_dim, 16)
            self.conv2 = GCNLayer_Factored_MPNN_1(16, output_dim)
        else:
            print('Standard implementation')
            self.conv1 = GCNLayer_MPNN_1(input_dim, 16)
            self.conv2 = GCNLayer_MPNN_1(16, output_dim)

    def forward(self, V, E, X):
        H = nn.functional.relu(self.conv1(V, E, X))
        H = nn.functional.dropout(H, training=self.training)
        H = self.conv2(V, E, H)
        return H # PyG uses log softmax here


class GCN_2_Paper(torch.nn.Module):
    def __init__(self, input_dim, output_dim, factored=False, forwards=False):
        super().__init__()
        assert (factored and forwards) == False, 'Factored and forwards not supported at the moment'
        if factored:
            print('Factored implementation')
            self.conv1 = GCNLayer_Factored_MPNN_2(input_dim, 16)
            self.conv2 = GCNLayer_Factored_MPNN_2(16, output_dim)
        elif forwards:
            print('Forwards implemenation')
            self.conv1 = GCNLayer_MPNN_2_Forwards(input_dim, 16)
            self.conv2 = GCNLayer_MPNN_2_Forwards(16, output_dim)
        else:
            print('Standard implementation')
            self.conv1 = GCNLayer_MPNN_2(input_dim, 16)
            self.conv2 = GCNLayer_MPNN_2(16, output_dim)

    def forward(self, V, E, X):
        H = nn.functional.relu(self.conv1(V, E, X))
        H = nn.functional.dropout(H, training=self.training)
        H = self.conv2(V, E, H)
        return H # PyG uses log softmax here


class PyG_GCN_Paper(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.conv1 = torch_geometric.nn.conv.GCNConv(input_dim, 16, cached=False, add_self_loops=True)
        self.conv2 = torch_geometric.nn.conv.GCNConv(16, output_dim, cached=False, add_self_loops=True)

    def forward(self, V, E, X):
        H = nn.functional.relu(self.conv1(X, E))
        H = nn.functional.dropout(H, training=self.training)
        H = self.conv2(H, E)
        return H # PyG uses log softmax here

