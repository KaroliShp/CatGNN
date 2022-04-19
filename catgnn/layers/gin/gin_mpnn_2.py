import torch
from torch import nn
import torch_scatter
import torch_geometric  # TODO: write our own reset function

from catgnn.integral_transform.mpnn_2 import BaseMPNNLayer_2


class GINLayer_MPNN_2(BaseMPNNLayer_2):
    def __init__(self, mlp_update, device, eps: float = 0.0, train_eps: bool = True):
        super().__init__()

        self.device = device
        self.eps_0 = eps
        self.train_eps = train_eps
        self.eps = nn.Parameter(torch.tensor([eps]), requires_grad=train_eps).to(device)
        self.mlp_update = mlp_update

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
            aggregated = torch_scatter.scatter_add(
                edge_messages.T, bag_indices.repeat(edge_messages.T.shape[0], 1)
            ).T
            return aggregated[V]

        return aggregator

    def update(self, X, output):
        return self.mlp_update(((1 + self.eps) * X) + output)

    def reset_parameters(self):
        # TODO: write our own reset function
        torch_geometric.nn.inits.reset(self.mlp_update)
        self.eps = nn.Parameter(
            torch.Tensor([self.eps_0]), requires_grad=self.train_eps
        ).to(self.device)