import torch
from torch import nn
import torch_scatter

from catgnn.integral_transform.mpnn_2 import BaseMPNNLayer_2


class SGCLayer_MPNN_2(BaseMPNNLayer_2):
    def __init__(self, in_dim: int, out_dim: int, K: int = 1):
        super().__init__()

        self.mlp_update = nn.Linear(in_dim, out_dim)  # \psi
        self.K = K

    def forward(self, V, E, X):
        # Add self-loops to the adjacency matrix.
        E = torch.cat((E, torch.arange(V.shape[0]).repeat(2, 1)), dim=1)

        # Compute normalization.
        self.degrees = torch.zeros(V.shape[0], dtype=torch.int64).scatter_add_(
            0, E[1], torch.ones(E.shape[1], dtype=torch.int64)
        )
        self.norm = torch.sqrt(1 / (self.degrees[E[0]] * self.degrees[E[1]]))

        # Do integral transform (just like PyG - not sure if this is the best way to do it?)
        # Why then not just always let K=1 and do this by stacking the layers? Why K as an argument?
        out = self.transform_backwards(V, E, X)
        for k in range(self.K - 1):
            out = self.transform_backwards(V, E, out)

        return self.mlp_update(out)

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
            aggregated = torch_scatter.scatter_add(
                edge_messages.T, bag_indices.repeat(edge_messages.T.shape[0], 1)
            ).T
            return aggregated[V]

        return aggregator

    def update(self, X, output):
        return output

    """
    Other methods (TODO)
    """

    def reset_parameters(self):
        self.mlp_update.reset_parameters()
