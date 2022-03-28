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
        #print(E)
        return E[:,1]

    def t(self, E: torch.Tensor) -> torch.Tensor:
        #print(E[:,0])
        return E[:,0]

    def _receiver_preimage_set(self, n_V, E):
        #print(E)
        degrees = torch.zeros(n_V, dtype=E.dtype).scatter_add_(0, E[0], torch.ones(E[0].shape, dtype=E.dtype))
        indices = torch.cumsum(degrees, dim=0)
        #print(f'degrees: {degrees}')
        #print(f'indices: {indices}')
        self._t_1_set = [E[:,:indices[0]].T]
        #print(self._t_1_set)
        for i in range(1,n_V):
            self._t_1_set.append(E[:,indices[i-1]:indices[i]].T)
            #print(self._t_1_set)
    
    def t_1(self, V: torch.Tensor) -> torch.Tensor:
        preimage = []
        for v in V:
            preimage.append(self._t_1_set[v])
        #print(f'PREIMAGE of {V}: {preimage}')
        return preimage

    """
    Other building blocks for implementing primitive operations
    """
    
    def f(self, V: torch.Tensor) -> torch.Tensor:
        return self.X[V.long()]

    """
    Integral transform primitives
    """

    def define_pullback(self, f: Type_V_R) -> Type_E_R:
        def pullback(E: torch.Tensor) -> torch.Tensor:
            raise NotImplementedError
        return pullback

    def define_kernel(self, er: Type_E_R) -> Type_E_R:
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

    def pipeline(self, V: List[Type_V], E: torch.Tensor, X: torch.Tensor):
        # Set the span diagram and feature function f : V -> R
        self._receiver_preimage_set(V.shape[0], E.T)        
        self.X = X

        # Prepare pipeline
        pullback = self.define_pullback(self.f) # E -> R
        kernel_transformation = self.define_kernel(pullback) # E -> R
        pushforward = self.define_pushforward(kernel_transformation) # V -> N[R]
        aggregator = self.define_aggregator(pushforward) # V -> R

        # Apply the pipeline to each node in the graph
        return aggregator(V)