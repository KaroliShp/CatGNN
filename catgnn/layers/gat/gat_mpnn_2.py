import torch
from torch import nn
import torch_scatter

from catgnn.integral_transform.mpnn_2 import BaseMPNNLayer_2
from catgnn.utils import add_self_loops


class GATLayer_MPNN_2(BaseMPNNLayer_2):
    def __init__(self, in_dim: int, out_dim: int, heads: int = 1):
        # Start with only 1 attention head for simplicity
        super().__init__()

        self.heads = heads
        self.mlp_msgs = []
        self.attention_as = []
        for _ in range(heads):
            self.mlp_msgs.append(nn.Linear(in_dim, out_dim))
            self.attention_as.append(nn.Linear(in_dim * 2, out_dim))
        self.mlp_msgs = nn.ModuleList(self.mlp_msgs)
        self.attention_as = nn.ModuleList(self.attention_as)

    def forward(self, V, E, X):
        # Add self-loops to the adjacency matrix
        # GAT paper: " In all our experiments, these will be exactly the first-order neighbors of i (including i)"
        E = add_self_loops(V, E)

        # TODO: add dropout within the layer?

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
            # Get features of both senders and receivers
            r_sender, r_receiver = kernel_factor_1(E)

            updated_features = None
            for h in range(self.heads):
                # Attention: calculate a_{i,j}
                concatenated_features = torch.cat(
                    (r_sender, r_receiver), -1
                )  # Note that we concatenate on the last dimension (-1) (rows)

                attention_coefficients = torch.nn.functional.leaky_relu(
                    self.attention_as[h](concatenated_features),
                    negative_slope=0.02
                ).exp()  # e_{i,j}

                softmax_denominator = torch_scatter.scatter_add(
                    attention_coefficients, self.t(E), dim=0
                )  # Must be same shape as X

                softmaxed_coefficients = (
                    attention_coefficients / softmax_denominator[self.t(E)]
                )

                if updated_features is None:
                    updated_features = softmaxed_coefficients * self.mlp_msgs[h](
                        r_sender
                    )
                else:
                    updated_features = torch.cat(
                        (
                            updated_features,
                            softmaxed_coefficients * self.mlp_msgs[h](r_sender),
                        ),
                        dim=-1,
                    )

            # Perform kernel transform
            return updated_features

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
        for m in self.mlp_msgs:
            m.reset_parameters()
        for a in self.attention_as:
            a.reset_parameters()


if __name__ == "__main__":
    V = torch.tensor([0, 1, 2, 3], dtype=torch.int64)

    # E is a set of edges - usual sparse representation in PyG
    E = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 1]], dtype=torch.int64)

    # Feature matrix - usual representation
    X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float)

    example_layer = GATLayer_MPNN_2(2, 2, heads=8)
    print(example_layer(V, E, X))
    example_layer.reset_parameters()
