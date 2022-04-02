import torch
from torch import nn
from catgnn.typing import *


class BaseMPNNLayer_2(nn.Module):

    def __init__(self):
        super(BaseMPNNLayer_2, self).__init__()

    """
    Directed graph span construction
    """

    def s(self, E: torch.Tensor) -> torch.Tensor:
        return E[1]

    def t(self, E: torch.Tensor) -> torch.Tensor:
        return E[0]

    def _set_preimages(self, n_V, E):
        degrees = torch.zeros(n_V, dtype=E.dtype).scatter_add_(0, E[0], torch.ones(E[0].shape, dtype=E.dtype))
        indices = torch.cumsum(degrees, dim=0)
        self._preimages = [E[:,:indices[0]]]
        for i in range(1,n_V):
            self._preimages.append(E[:,indices[i-1]:indices[i]])
    
    def t_1(self, V: torch.Tensor) -> torch.Tensor:
        preimage = []
        for v in V:
            preimage.append(self._preimages[v])
        return preimage
    
    def f(self, V: torch.Tensor) -> torch.Tensor:
        return self.X[V]

    """
    Integral transform primitives
    """

    def define_pullback(self, f: Type_V_R) -> Type_E_R:
        def pullback(E: torch.Tensor) -> torch.Tensor:
            raise NotImplementedError
        return pullback

    def define_kernel_factor_1(self, pullback):
        def kernel_factor_1(E, E_star):
            raise NotImplementedError
        return kernel_factor_1

    def define_kernel_factor_2(self, kernel_factor_1):
        def kernel_factor_2(E):
            raise NotImplementedError
        return kernel_factor_2

    def define_kernel(self, pullback: Type_E_R) -> Type_E_R:
        def kernel_transformation(E: torch.Tensor) -> torch.Tensor:
            raise NotImplementedError
        return kernel_transformation

    def define_pushforward(self, kernel_transformation: Type_E_R) -> Type_V_NR:
        def pushforward(V: List[Type_V]) -> Type_NR:
            raise NotImplementedError
        return pushforward
    
    def define_aggregator(self, pushforward: Type_V_NR) -> Type_V_R:
        def aggregator(V: List[Type_V]) -> torch.Tensor:
            raise NotImplementedError
        return aggregator

    def update(self, output):
        raise NotImplementedError

    def pipeline(self, V: torch.Tensor, E: torch.Tensor, X: torch.Tensor, kernel_factor=False):
        # Set the span diagram and feature function f : V -> R
        self._set_preimages(V.shape[0], E)        
        self.X = X

        # Prepare pipeline
        pullback = self.define_pullback(self.f) # E -> R
        if kernel_factor:
            product_arrow = self.define_kernel_factor_1(pullback) # (E -> R) x (E -> R)
            kernel_transformation = self.define_kernel_factor_2(product_arrow) # E -> R
        else:
            kernel_transformation = self.define_kernel(pullback) # E -> R
        pushforward = self.define_pushforward(kernel_transformation) # V -> N[R]
        aggregator = self.define_aggregator(pushforward) # V -> R

        # Apply the pipeline to each node in the graph
        return self.update(aggregator(V))