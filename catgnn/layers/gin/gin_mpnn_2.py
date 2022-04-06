from catgnn.integral_transform.mpnn_2 import BaseMPNNLayer_2
from catgnn.typing import *
import torch
from torch import nn
import torch_scatter


class GINLayer_MPNN_2(BaseMPNNLayer_2):

    def __init__(self, mlp_update, eps: float=0.0):
        super().__init__()

        self.eps = nn.Parameter(torch.Tensor([eps]), requires_grad=True)
        self.mlp_update = mlp_update
    
    def forward(self, V: torch.Tensor, E: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        # Do integral transform
        return self.pipeline_backwards(V, E, X)

    def define_pullback(self, f):
        def pullback(E):
            return f(self.s(E))
        return pullback

    def define_kernel(self, pullback):
        def kernel(E):
            return pullback(E)
        return kernel

    def define_pushforward(self, kernel):
        def pushforward(V, E):
            # Need to call preimage here
            return kernel(E), self.t(E)
        return pushforward

    def define_aggregator(self, pushforward):
        def aggregator(V, E):
            bags = pushforward(V,E)
            aggregated = torch_scatter.scatter_add(bags[0].T, bags[1].repeat(bags[0].T.shape[0],1)).T
            return aggregated[V]
        return aggregator

    def update(self, X, output):
        return self.mlp_update(((1+self.eps)*X) + output)