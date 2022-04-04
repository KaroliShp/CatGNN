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

    def t_1(self, V):
        # Get preimages of only those edges where the receiver is in V
        # For now, assume V is always all nodes (we cannot select the nodes)
        return self.E, self.t(self.E)
    
    def f(self, V: torch.Tensor) -> torch.Tensor:
        return self.X[V]

    """
    Integral transform primitives (backwards)
    """

    def define_pullback(self, f):
        def pullback(E):
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

    def define_kernel(self, pullback):
        def kernel(E):
            raise NotImplementedError
        return kernel
    
    def define_pushforward(self, kernel):
        def pushforward(V):
            raise NotImplementedError
        return pushforward
    
    def define_aggregator(self, pushforward):
        def aggregator(V):
            raise NotImplementedError
        return aggregator

    def update(self, X, output):
        raise NotImplementedError

    def pipeline_backwards(self, V: torch.Tensor, E: torch.Tensor, X: torch.Tensor, kernel_factor=False):
        # Set the span diagram and feature function f : V -> R
        self.X = X
        self.E = E

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
        return self.update(X, aggregator(V))

    """
    Integral transform primitives (forwards)
    """

    def pullback(self, E, f):
        raise NotImplementedError

    def kernel_factor_1(self, E, E_star, pulledback_features):
        raise NotImplementedError

    def kernel_factor_2(self, E, kernel_factor_1):
        raise NotImplementedError

    def kernel_transformation(self, E, pulledback_features):
        raise NotImplementedError

    def pushforward(self, V, E, edge_messages):
        raise NotImplementedError
    
    def aggregator(self, V, bags_of_values):
        raise NotImplementedError

    def update(self, X, output):
        raise NotImplementedError

    def pipeline_forwards(self, V: torch.Tensor, E: torch.Tensor, X: torch.Tensor, kernel_factor=False):    
        self.X = X

        # Execute pipeline
        pulledback_features = self.pullback(E, self.f)
        #print(f'pulledback features: {pulledback_features}\n')
        edge_messages = self.kernel_transformation(E, pulledback_features)
        #print(f'edge messages: {edge_messages}\n')
        bags_of_values = self.pushforward(V, E, edge_messages)
        #print(f'bags of values: {bags_of_values}\n')
        output = self.aggregator(V, bags_of_values)
        #print(f'output: {output}')
        return self.update(X, output)