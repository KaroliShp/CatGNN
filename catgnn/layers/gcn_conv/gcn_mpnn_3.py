from catgnn.integral_transform.mpnn_3 import BaseMPNNLayer_3
from catgnn.typing import *
import torch
from torch import nn
import torch_scatter


class GCNLayer_MPNN_3(BaseMPNNLayer_3):

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()

        self.mlp_msg = nn.Linear(in_dim, out_dim) # \psi
        self.mlp_update = nn.LeakyReLU() # \phi
    
    def forward(self, V: torch.Tensor, E: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        # Compute degrees
        self.degrees = torch.zeros(V.shape[0], dtype=E.dtype).scatter_add_(0, E.T[0], torch.ones(E.T[0].shape, dtype=E.dtype)) + 1
        
        # Add self-loops
        E = torch.cat((E,torch.arange(V.shape[0]).repeat(2,1)), dim=1)

        # 3. Compute normalization and provide as edge features for kernel transform
        self.norm = torch.sqrt(1/self.degrees[E[0]] * self.degrees[E[1]])

        # Do integral transform
        return self.pipeline_backwards(V, E, X)

    def define_pullback(self, f):
        def pullback(E):
            return f(self.s(E))
        return pullback

    def define_kernel(self, pullback):
        def kernel(E):
            return self.mlp_msg(pullback(E)) * self.norm.view(-1, 1)
        return kernel

    def define_pushforward(self, kernel):
        def pushforward(V):
            # Need to call preimage here
            E, indices = self.t_1(V)
            return kernel(E), indices
        return pushforward

    def define_aggregator(self, pushforward):
        def aggregator(V):
            bags = pushforward(V)
            aggregated = torch_scatter.scatter_add(bags[0].T, bags[1].repeat(bags[0].T.shape[0],1)).T
            return aggregated[V]
        return aggregator

    def update(self, X, output):
        return self.mlp_update(output)


class GCNLayer_MPNN_3_Forwards(BaseMPNNLayer_3):

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()

        self.mlp_msg = nn.Linear(in_dim, out_dim) # \psi
        self.mlp_update = nn.LeakyReLU() # \phi
    
    def forward(self, V: torch.Tensor, E: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        # Compute degrees
        self.degrees = torch.zeros(V.shape[0], dtype=E.dtype).scatter_add_(0, E.T[0], torch.ones(E.T[0].shape, dtype=E.dtype)) + 1
        
        # Add self-loops
        E = torch.cat((E,torch.arange(V.shape[0]).repeat(2,1)), dim=1)

        # 3. Compute normalization and provide as edge features for kernel transform
        self.norm = torch.sqrt(1/self.degrees[E[0]] * self.degrees[E[1]])

        # Do integral transform
        return self.pipeline_forwards(V, E, X)

    def pullback(self, E, f):
        return f(self.s(E))

    def kernel_transformation(self, E, pulledback_features):
        return self.mlp_msg(pulledback_features) * self.norm.view(-1, 1)

    def pushforward(self, V, E, edge_messages):
        return edge_messages, self.t(E)
    
    def aggregator(self, V, bags_of_values):
        aggregated = torch_scatter.scatter_add(bags_of_values[0].T, 
                                               bags_of_values[1].repeat(bags_of_values[0].T.shape[0],1)).T
        return aggregated[V]

    def update(self, X, output):
        return self.mlp_update(output)