import torch

from catgnn.integral_transform.mpnn_1 import BaseMPNNLayer_1
from catgnn.utils import add_self_loops, get_degrees


class SGCLayer_MPNN_1(BaseMPNNLayer_1):
    def __init__(self, in_dim: int, out_dim: int, K: int = 1):
        super().__init__()

        self.mlp_update = torch.nn.Linear(in_dim, out_dim)  # \psi
        self.K = K

    def forward(self, V, E, X):
        # Add self-loops to the adjacency matrix.
        # E = torch.cat((E,torch.arange(V.shape[0]).repeat(2,1)), dim=1)
        E = add_self_loops(V, E)

        # Compute normalization.
        # self.degrees = torch.zeros(V.shape[0], dtype=torch.int64).scatter_add_(0, E[1], torch.ones(E.shape[1], dtype=torch.int64))
        self.degrees = get_degrees(V, E)
        self.norm = torch.sqrt(1 / (self.degrees[E[0]] * self.degrees[E[1]]))

        # Do integral transform (just like PyG - not sure if this is the best way to do it?)
        # Why then not just always let K=1 and do this by stacking the layers? Why K as an argument?
        out = self.transform_backwards(V, E, X)
        for k in range(self.K - 1):
            out = self.transform_backwards(V, E, out)

        return self.mlp_update(out)

    def define_pullback(self, f):
        def pullback(e):
            return f(self.s(e))

        return pullback

    def define_kernel(self, pullback):
        def kernel_transformation(e):
            return pullback(e) * self.norm[e[2]]

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

    """
    Other methods (TODO)
    """

    def reset_parameters(self):
        self.mlp_update.reset_parameters()


if __name__ == "__main__":
    # Example graph above
    # V is a set of nodes - usual representation
    V = torch.tensor([0, 1, 2, 3], dtype=torch.int64)

    # E is a set of edges - usual sparse representation in PyG
    E = torch.tensor(
        [(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)], dtype=torch.int64
    ).T

    # Feature matrix - usual representation
    X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)

    example_layer = SGCLayer_MPNN_1(2, 2)
    print(example_layer(V, E, X))
