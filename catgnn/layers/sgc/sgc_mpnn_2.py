import torch
from torch import nn
import torch_scatter

from catgnn.integral_transform.mpnn_2 import BaseMPNNLayer_2
from catgnn.utils import add_self_loops, get_degrees


class SGCLayer_MPNN_2(BaseMPNNLayer_2):
    """
    SGC layer using standard (backwards) implementation with BaseMPNNLayer_2

    Args:
        in_dim (int): input dimension for the message linear layer
        out_dim (int): output dimension for the message linear layer
        K (int): K from the paper (see code comments for usage)
    """

    def __init__(self, in_dim: int, out_dim: int, K: int = 1):
        super().__init__()

        self.mlp_update = nn.Linear(in_dim, out_dim)  # \psi
        self.K = K

    def forward(self, V, E, X):
        # Add self-loops to the adjacency matrix.
        E = add_self_loops(V, E)

        # Compute normalization as edge weights
        self.degrees = get_degrees(V, E)
        self.edge_weights = torch.sqrt(1 / (self.degrees[self.s(E)] * self.degrees[self.t(E)]))

        # Do integral transform (this is done like in PyG - not sure if this is the best way to do it?)
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
            return pullback(E) * self.edge_weights.view(-1, 1)

        return kernel

    def define_pushforward(self, kernel):
        def pushforward(V):
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

    def reset_parameters(self):
        self.mlp_update.reset_parameters()
