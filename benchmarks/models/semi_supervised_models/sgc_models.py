from benchmarks.utils.train_semi_supervised import train_eval_loop
from catgnn.layers.gcn.gcn_mpnn_1 import GCNLayer_MPNN_1, GCNLayer_Factored_MPNN_1
from catgnn.layers.gcn.gcn_mpnn_2 import GCNLayer_MPNN_2, GCNLayer_Factored_MPNN_2, GCNLayer_MPNN_2_Forwards
from catgnn.layers.sgc.sgc_mpnn_2 import SGCLayer_MPNN_2
import torch
from torch import nn
from catgnn.datasets.planetoid import PlanetoidDataset
import torch_geometric


class SGC_2(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(SGC_2, self).__init__()
        self.gcn_layer = SGCLayer_MPNN_2(input_dim, output_dim, K=2)

    def forward(self, V, E, X):
        return self.gcn_layer(V, E, X)


class PyG_SGC(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(PyG_SGC, self).__init__()
        self.gcn_layer = torch_geometric.nn.conv.SGConv(input_dim, output_dim, K=2, cached=False, add_self_loops=True)

    def forward(self, V, E, X):
        return self.gcn_layer(X, E)
