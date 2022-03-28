from catgnn.integral_transform.mpnn_3 import BaseMPNNLayer_3
from catgnn.typing import *
import torch
from torch import nn
import numpy as np
from torch_geometric.utils import add_self_loops, degree # TODO - add our own
import torch_scatter


class GCNLayer_MPNN_3(BaseMPNNLayer_3):

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()

        self.mlp_msg = nn.Linear(in_dim, out_dim) # \psi
        self.mlp_update = nn.LeakyReLU() # \phi
    
    def forward(self, V: List[Type_V], E: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        # Add self-loops
        E = torch.cat((E,torch.arange(V.shape[0]).repeat(2,1)), dim=1)

        # Step 3: Compute normalization.
        row, col = E
        deg = degree(col, X.size(0), dtype=X.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        self.norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        #print(norm.shape)

        out = self.pipeline(V, E, X)
        return self.mlp_update(out)

    def define_pullback(self, f):
        def pullback(E):
            return f(self.s(E))
        return pullback

    def define_kernel(self, pullback):
        def kernel(E):
            return self.mlp_msg(pullback(E)) * self.norm.view(-1, 1)
        return kernel

    def define_pushforward(self, kernel):
        def pushforward(V, E):
            return kernel(E), self.t(E)
        return pushforward

    def define_aggregator(self, pushforward):
        def aggregator(V, E):
            bags = pushforward(V,E)
            aggregated = torch_scatter.scatter_add(bags[0].T, bags[1].repeat(bags[0].T.shape[0],1)).T
            return aggregated[V]
        return aggregator