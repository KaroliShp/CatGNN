import torch
from torch import nn
from catgnn.typing import *


class BaseMPNNLayer_3(nn.Module):
    def __init__(self):
        super(BaseMPNNLayer_3, self).__init__()

    """
    Directed graph span construction
    """

    def s(self, E: torch.Tensor) -> torch.Tensor:
        return E[1]

    def t(self, E: torch.Tensor) -> torch.Tensor:
        return E[0]

    """
    Other building blocks for implementing primitive operations
    """
    
    def f(self, V: torch.Tensor) -> torch.Tensor:
        return self.X[V]

    """
    Primitives
    """

    def define_pullback(self, f):
        def pullback(E):
            raise NotImplementedError
        return pullback
    
    def define_kernel(self, pullback):
        def kernel(E):
            raise NotImplementedError
        return kernel
    
    def define_pushforward(self, kernel):
        def pushforward(V, E):
            raise NotImplementedError
        return pushforward
    
    def define_aggregator(self, pushforward):
        def aggregator(V):
            raise NotImplementedError
        return aggregator

    def update(self, output):
        raise NotImplementedError

    def pipeline(self, V: torch.Tensor, E: torch.Tensor, X: torch.Tensor):
        # Set the span diagram and feature function f : V -> R
        self.X = X

        # Prepare pipeline
        pullback = self.define_pullback(self.f) # E -> R
        kernel = self.define_kernel(pullback) # E -> R
        pushforward = self.define_pushforward(kernel) # V -> N[R]
        aggregator = self.define_aggregator(pushforward) # V -> R

        # Apply the pipeline to each node in the graph
        return self.update(aggregator(V,E))