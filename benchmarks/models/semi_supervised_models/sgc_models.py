from catgnn.layers.sgc.sgc_mpnn_2 import SGCLayer_MPNN_2
import torch
from torch import nn
import torch_geometric


"""
General layers
"""


class SGC_2(nn.Module):
    def __init__(self, input_dim, output_dim, K=2):
        super(SGC_2, self).__init__()
        self.gcn_layer = SGCLayer_MPNN_2(input_dim, output_dim, K=K)

    def forward(self, V, E, X):
        return self.gcn_layer(V, E, X)


class PyG_SGC(nn.Module):
    def __init__(self, input_dim, output_dim, K=2):
        super(PyG_SGC, self).__init__()
        self.gcn_layer = torch_geometric.nn.conv.SGConv(
            input_dim, output_dim, K=K, cached=False, add_self_loops=True
        )

    def forward(self, V, E, X):
        return self.gcn_layer(X, E)


"""
Paper benchmarking
"""


class SGC_2_Paper(torch.nn.Module):
    def __init__(self, input_dim, output_dim, K=3):
        super().__init__()
        self.conv1 = SGCLayer_MPNN_2(input_dim, output_dim, K=K)

    def forward(self, V, E, X):
        H = self.conv1(V, E, X)
        return H


class PyG_SGC_Paper(torch.nn.Module):
    def __init__(self, input_dim, output_dim, K=3):
        super().__init__()
        self.conv1 = torch_geometric.nn.conv.SGConv(
            input_dim, output_dim, K=K, cached=False, add_self_loops=True
        )

    def forward(self, V, E, X):
        H = self.conv1(X, E)
        return H
