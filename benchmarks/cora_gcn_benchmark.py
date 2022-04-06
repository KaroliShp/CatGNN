from benchmarks.utils.train_transductive import train_eval_loop
from catgnn.layers.gcn_conv.gcn_mpnn_1 import GCNLayer_MPNN_1, GCNLayer_Factored_MPNN_1
from catgnn.layers.gcn_conv.gcn_mpnn_2 import GCNLayer_MPNN_2, GCNLayer_Factored_MPNN_2, GCNLayer_MPNN_2_Forwards
from catgnn.layers.sgc.sgc_mpnn_2 import SGCLayer_MPNN_2
import torch
from torch import nn
from catgnn.datasets.planetoid import PlanetoidDataset
import torch_geometric


"""
GCN models
"""


class CatGNN_GCN_1(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(CatGNN_GCN_1, self).__init__()
        self.gcn_layer = GCNLayer_MPNN_1(input_dim, output_dim)

    def forward(self, V, E, X):
        return self.gcn_layer(V, E, X)


class CatGNN_Factored_GCN_1(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(CatGNN_Factored_GCN_1, self).__init__()
        self.gcn_layer = GCNLayer_Factored_MPNN_1(input_dim, output_dim)

    def forward(self, V, E, X):
        return self.gcn_layer(V, E, X)


class CatGNN_GCN_2(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(CatGNN_GCN_2, self).__init__()
        self.gcn_layer = GCNLayer_MPNN_2(input_dim, output_dim)

    def forward(self, V, E, X):
        return self.gcn_layer(V, E, X)


class CatGNN_Factored_GCN_2(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(CatGNN_Factored_GCN_2, self).__init__()
        self.gcn_layer = GCNLayer_Factored_MPNN_2(input_dim, output_dim)

    def forward(self, V, E, X):
        return self.gcn_layer(V, E, X)


class CatGNN_GCN_2_Forwards(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(CatGNN_GCN_2_Forwards, self).__init__()
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
SGC models
"""


class CatGNN_SGC_2(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(CatGNN_SGC_2, self).__init__()
        self.gcn_layer = SGCLayer_MPNN_2(input_dim, output_dim, K=2)

    def forward(self, V, E, X):
        return self.gcn_layer(V, E, X)


class PyG_SGC(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(PyG_SGC, self).__init__()
        self.gcn_layer = torch_geometric.nn.conv.SGConv(input_dim, output_dim, K=2, cached=False, add_self_loops=True)

    def forward(self, V, E, X):
        return self.gcn_layer(X, E)


"""
Benchmark methods
"""


def benchmark(model_nn, sender_to_receiver=True):
    """
    Benchmark Cora dataset
    """
    cora_dataset = PlanetoidDataset('Cora')

    train_x, train_y, val_x, val_y, test_x, test_y = cora_dataset.train_val_test_split()
    train_mask, val_mask, test_mask = cora_dataset.get_split_masks()

    V = cora_dataset.get_vertices()
    E = cora_dataset.get_edges(sender_to_receiver)
    X = cora_dataset.get_features()

    model = model_nn(input_dim=train_x.shape[-1], output_dim=7)

    train_stats_gnn_cora = train_eval_loop(model, 
                                           V, E, X, train_y, train_mask, 
                                           V, E, X, val_y, val_mask, 
                                           V, E, X, test_y, test_mask)
    print(train_stats_gnn_cora)


def benchmark_catgnn_gcn_1():
    benchmark(CatGNN_GCN_1, sender_to_receiver=False)


def benchmark_catgnn_factored_gcn_1():
    benchmark(CatGNN_Factored_GCN_1, sender_to_receiver=False)


def benchmark_catgnn_gcn_2():
    benchmark(CatGNN_GCN_2, sender_to_receiver=False)


def benchmark_catgnn_factored_gcn_2():
    benchmark(CatGNN_Factored_GCN_2, sender_to_receiver=False)


def benchmark_catgnn_gcn_2_forwards():
    benchmark(CatGNN_GCN_2_Forwards, sender_to_receiver=False)


def benchmark_catgnn_sgc_2():
    benchmark(CatGNN_SGC_2, sender_to_receiver=False)


def benchmark_pyg_gcn():
    benchmark(PyG_GCN)


def benchmark_pyg_sgc():
    benchmark(PyG_SGC)


if __name__ == '__main__':
    """
    GCN
    """
    #benchmark_catgnn_gcn_1()
    #benchmark_catgnn_factored_gcn_1()

    #benchmark_catgnn_gcn_2()
    #benchmark_catgnn_factored_gcn_2()
    #benchmark_catgnn_gcn_2_forwards()

    #benchmark_pyg_gcn()

    """
    SGC
    """
    #benchmark_catgnn_sgc_2()

    benchmark_pyg_sgc()