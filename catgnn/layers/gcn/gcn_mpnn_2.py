import torch
from torch import nn
import torch_scatter

from catgnn.integral_transform.mpnn_2 import BaseMPNNLayer_2
from catgnn.utils import add_self_loops, get_degrees


class GCNLayer_MPNN_2(BaseMPNNLayer_2):
    """
    GCN layer using standard (backwards) implementation with BaseMPNNLayer_2

    Args:
        in_dim (int): input dimension for the message linear layer
        out_dim (int): output dimension for the message linear layer
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()

        self.mlp_msg = nn.Linear(in_dim, out_dim, bias=False)  # \psi
        self.mlp_update = nn.LeakyReLU()  # \phi

    def forward(self, V, E, X):
        # Add self-loops to the adjacency matrix.
        E = add_self_loops(V, E)

        # Compute normalization as edge weights
        self.degrees = get_degrees(V, E)
        self.edge_weights = torch.sqrt(1 / (self.degrees[E[0]] * self.degrees[E[1]]))

        # Do integral transform
        return self.transform_backwards(V, E, X)

    def define_pullback(self, f):
        def pullback(E):
            return f(self.s(E))

        return pullback

    def define_kernel(self, pullback):
        def kernel(E):
            return self.edge_weights.view(-1, 1) * self.mlp_msg(pullback(E))

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
        # Not sure if better to leave this as this and add update in the actual model (like PyG)
        # or to add the update here (same thing applies to all other layers)
        return output

    def reset_parameters(self):
        self.mlp_msg.reset_parameters()


class GCNLayer_Factored_MPNN_2(BaseMPNNLayer_2):
    """
    GCN layer using standard (backwards) implementation with BaseMPNNLayer_2 
    and unnecessary factoring of the kernel arrow, i.e. receiver features are
    simply ignored

    Args:
        in_dim (int): input dimension for the message linear layer
        out_dim (int): output dimension for the message linear layer
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()

        self.mlp_msg = nn.Linear(in_dim, out_dim, bias=False)  # \psi
        self.mlp_update = nn.LeakyReLU()  # \phi

    def forward(self, V, E, X):
        # Add self-loops to the adjacency matrix.
        E = add_self_loops(V, E)

        # Compute normalization as edge weights
        self.degrees = get_degrees(V, E)
        self.edge_weights = torch.sqrt(1 / (self.degrees[E[0]] * self.degrees[E[1]]))

        # Do integral transform
        return self.transform_backwards(V, E, X, kernel_factor=True)

    def define_pullback(self, f):
        def pullback(E):
            return f(self.s(E))

        return pullback

    def define_kernel_factor_1(self, pullback):
        def kernel_factor_1(E):
            E_star = self.get_opposite_edges(E)
            return pullback(E), pullback(E_star)

        return kernel_factor_1

    def define_kernel_factor_2(self, kernel_factor_1):
        def kernel_factor_2(E):
            r_sender, r_receiver = kernel_factor_1(E)
            return self.edge_weights.view(-1, 1) * self.mlp_msg(r_sender)

        return kernel_factor_2

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
        self.mlp_msg.reset_parameters()


class GCNLayer_MPNN_2_Forwards(BaseMPNNLayer_2):
    """
    GCN layer using forwards implementation with BaseMPNNLayer_2

    Args:
        in_dim (int): input dimension for the message linear layer
        out_dim (int): output dimension for the message linear layer
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()

        self.mlp_msg = nn.Linear(in_dim, out_dim, bias=False)  # \psi
        self.mlp_update = nn.LeakyReLU()  # \phi

    def forward(self, V, E, X):
        # Add self-loops to the adjacency matrix.
        E = add_self_loops(V, E)

        # Compute normalization as edge weights
        self.degrees = get_degrees(V, E)
        self.edge_weights = torch.sqrt(1 / (self.degrees[E[0]] * self.degrees[E[1]]))

        # Do integral transform
        return self.transform_forwards(V, E, X)

    def pullback(self, E, f):
        return f(self.s(E)), E

    def kernel(self, E, pulledback_features):
        return self.edge_weights.view(-1, 1) * self.mlp_msg(pulledback_features)

    def pushforward(self, V, edge_messages):
        E, bag_indices = self.t_1(V)
        return edge_messages, bag_indices

    def aggregator(self, V, edge_messages, bag_indices):
        aggregated = torch_scatter.scatter_add(
            edge_messages.T, bag_indices.repeat(edge_messages.T.shape[0], 1)
        ).T
        return aggregated[V]

    def update(self, X, output):
        return output

    def reset_parameters(self):
        self.mlp_msg.reset_parameters()
