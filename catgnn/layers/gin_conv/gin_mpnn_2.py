from catgnn.integral_transform.mpnn_2 import BaseMPNNLayer_2
from catgnn.typing import *
import torch
from torch import nn


class GINLayer_MPNN_2(BaseMPNNLayer_2):

    def __init__(self, mlp_update, eps: float=0.0):
        super().__init__()

        self.eps = nn.Parameter(torch.Tensor([eps]), requires_grad=True)
        self.mlp_update = mlp_update
    
    def forward(self, V: torch.Tensor, E: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        # Do integral transform
        return self.pipeline(V, E, X)

    def define_pullback(self, f: Type_V_R) -> Type_E_R:
        def pullback(E: torch.Tensor) -> torch.Tensor:
            return f(self.s(E))
        return pullback
    
    def define_kernel(self, pullback: Type_E_R) -> Type_E_R:
        def kernel_transformation(E: torch.Tensor) -> torch.Tensor:
            return pullback(E)
        return kernel_transformation
    
    def define_pushforward(self, kernel_transformation: Type_E_R) -> Type_V_NR:
        def pushforward(V: torch.Tensor) -> torch.Tensor:
            pE = self.t_1(V)
            bag = []
            for e in pE:
                bag.append(kernel_transformation(e))
            return bag
        
        return pushforward

    def define_aggregator(self, pushforward: Type_V_NR) -> Type_V_R:
        def aggregator(V: torch.Tensor) -> torch.Tensor:
            bag = pushforward(V)
            total = torch.Tensor()
            for b in bag:
                total = torch.hstack((total, torch.sum(b, dim=0)))
            return total.reshape(V.shape[0],-1)
        
        return aggregator

    def update(self, X, output):
        return self.mlp_update(((1+self.eps)*X) + output)