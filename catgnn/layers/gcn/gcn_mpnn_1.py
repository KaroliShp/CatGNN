import torch
from torch import nn

from catgnn.integral_transform.mpnn_1 import BaseMPNNLayer_1
from catgnn.utils import add_self_loops, get_degrees


class GCNLayer_MPNN_1(BaseMPNNLayer_1):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()

        self.mlp_msg = nn.Linear(in_dim, out_dim)  # \psi
        self.mlp_update = nn.LeakyReLU()  # \phi

    def forward(self, V, E, X):
        # Add self-loops to the adjacency matrix.
        E = add_self_loops(V, E)

        # Compute normalization as edge weights
        self.degrees = get_degrees(V, E)
        self.norm = torch.sqrt(1 / (self.degrees[E[0]] * self.degrees[E[1]]))

        # Do integral transform
        return self.transform_backwards(V, E, X)

    def define_pullback(self, f):
        def pullback(e):
            return f(self.s(e))

        return pullback

    def define_kernel(self, pullback):
        def kernel_transformation(e):
            return self.norm[e[2]] * self.mlp_msg(pullback(e))

        return kernel_transformation

    def define_pushforward(self, kernel_transformation):
        def pushforward(v):
            pE = self.t_1(v)

            # Now we need to apply edge_messages function for each element of pE
            bag_of_messages = []
            for e in pE:
                bag_of_messages.append(kernel_transformation(e))
            return bag_of_messages

        return pushforward

    def define_aggregator(self, pushforward):
        def aggregator(v):
            total = 0
            for val in pushforward(v):
                total += val
            return total

        return aggregator

    def update(self, X, output):
        return output

    def reset_parameters(self):
        self.mlp_msg.reset_parameters()
