from catgnn.integral_transform.mpnn_2 import BaseMPNNLayer_2
from catgnn.typing import *
import torch
from torch import nn
import torch_scatter


class SGCLayer_MPNN_2(BaseMPNNLayer_2):

    def __init__(self, in_dim: int, out_dim: int, K: int = 2):
        super().__init__()

        self.mlp_update = nn.Linear(in_dim, out_dim) # \psi
        self.K = K
    
    def forward(self, V: torch.Tensor, E: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        # Compute degrees
        self.degrees = torch.zeros(V.shape[0], dtype=E.dtype).scatter_add_(0, E.T[0], torch.ones(E.T[0].shape, dtype=E.dtype)) + 1
        
        # Add self-loops
        E = torch.cat((E,torch.arange(V.shape[0]).repeat(2,1)), dim=1)

        # 3. Compute normalization and provide as edge features for kernel transform
        self.norm = torch.sqrt(1/self.degrees[E[0]] * self.degrees[E[1]])

        out = self.pipeline_backwards(V, E, X)
        for k in range(self.K-1):
            out = self.pipeline_backwards(V, E, out)

        # Do integral transform
        return self.mlp_update(X)

    def define_pullback(self, f):
        def pullback(E):
            return f(self.s(E))
        return pullback

    def define_kernel(self, pullback):
        def kernel(E):
            return pullback(E) * self.norm.view(-1, 1)
        return kernel

    def define_pushforward(self, kernel):
        def pushforward(V):
            # Need to call preimage here
            E, bag_indices = self.t_1(V)
            return kernel(E), bag_indices
        return pushforward

    def define_aggregator(self, pushforward):
        def aggregator(V):
            edge_messages, bag_indices = pushforward(V)
            aggregated = torch_scatter.scatter_add(edge_messages.T, bag_indices.repeat(edge_messages.T.shape[0],1)).T
            return aggregated[V]
        return aggregator

    def update(self, X, output):
        return output

