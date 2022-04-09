from benchmarks.utils.train_semi_supervised import train_eval_loop
from catgnn.layers.gat.gat_mpnn_2 import GATLayer_MPNN_2
import torch
from torch import nn
import torch_geometric


"""
Paper benchmarking (architecture choice from PyG)
"""


class GAT_2_Paper(torch.nn.Module):
    def __init__(self, input_dim, output_dim, forwards=False):
        super().__init__()
        if forwards:
            raise NotImplementedError
        else:
            self.conv1 = GATLayer_MPNN_2(input_dim, 8)
            self.conv2 = GATLayer_MPNN_2(8, output_dim)

    def forward(self, V, E, X):
        H = nn.functional.dropout(X, p=0.6, training=self.training)
        H = nn.functional.elu(self.conv1(V, E, H))
        H = nn.functional.dropout(H, p=0.6, training=self.training)
        H = self.conv2(V, E, H)
        return H # PyG uses log softmax here

    def __repr__(self):
        return self.__class__.__name__


class PyG_GAT_Paper(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.conv1 = torch_geometric.nn.conv.GATConv(input_dim, 8, heads=1, add_self_loops=True)
        self.conv2 = torch_geometric.nn.conv.GATConv(8, output_dim, heads=1, concat=False, add_self_loops=True)

    def forward(self, V, E, X):        
        x = nn.functional.dropout(X, p=0.6, training=self.training)
        x = nn.functional.elu(self.conv1(x, E))
        x = nn.functional.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, E)
        return x # PyG uses log softmax here

    def __repr__(self):
        return self.__class__.__name__
