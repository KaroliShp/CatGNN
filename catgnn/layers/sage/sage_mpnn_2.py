from torch import nn
import torch_scatter

from catgnn.integral_transform.mpnn_2 import BaseMPNNLayer_2


class SAGELayer_MPNN_2(BaseMPNNLayer_2):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()

        self.mlp_update_1 = nn.Linear(in_dim, out_dim)
        self.mlp_update_2 = nn.Linear(in_dim, out_dim)

    def forward(self, V, E, X):
        # Do integral transform
        return self.transform_backwards(V, E, X)

    def define_pullback(self, f):
        def pullback(E):
            return f(self.s(E))

        return pullback

    def define_kernel(self, pullback):
        def kernel(E):
            return pullback(E)

        return kernel

    def define_pushforward(self, kernel):
        def pushforward(V):
            E, bag_indices = self.t_1(V)
            return kernel(E), bag_indices

        return pushforward

    def define_aggregator(self, pushforward):
        def aggregator(V):
            edge_messages, bag_indices = pushforward(V)
            aggregated = torch_scatter.scatter_mean(
                edge_messages.T, bag_indices.repeat(edge_messages.T.shape[0], 1)
            ).T
            return aggregated[V]

        return aggregator

    def update(self, X, output):
        return self.mlp_update_2(X) + self.mlp_update_1(output)

    """
    Other methods (TODO)
    """

    def reset_parameters(self):
        self.mlp_update_1.reset_parameters()
        self.mlp_update_2.reset_parameters()
