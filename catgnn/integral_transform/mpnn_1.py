import torch
from torch import nn
from catgnn.typing import *


class BaseMPNNLayer_1(nn.Module):

    def __init__(self):
        super(BaseMPNNLayer_1, self).__init__()

    """
    Directed graph span construction
    """

    def _add_edge_indices(self, E: torch.Tensor) -> torch.Tensor:
        return torch.cat((E, torch.arange(0, E.shape[1]).view(1,-1)))

    def s(self, e: torch.Tensor) -> torch.Tensor:
        return self.E_indexed[1][e[2]]

    def t(self, e: torch.Tensor) -> torch.Tensor:
        return self.E_indexed[0][e[2]]

    def _set_preimages(self, V):
        self._preimages = []

        # Create a list of lists, alternatively could be a dict
        for _ in V:
            self._preimages.append([])

        # Fill in all preimages with receiver and edge index
        for i in range(0, self.E_indexed.shape[1]):
            self._preimages[self.E_indexed[0][i]].append(self.E_indexed[:,i])

    def t_1(self, v: torch.Tensor) -> List[torch.Tensor]:
        return self._preimages[v]

    def f(self, v: torch.Tensor) -> torch.Tensor:
        return self.X[v]


    """
    Integral transform primitives
    """

    def define_pullback(self, f: Type_V_R) -> Type_E_R:
        def pullback(e: Type_E) -> Type_R:
            raise NotImplementedError
        return pullback

    def define_kernel(self, pullback: Type_E_R) -> Type_E_R:
        def kernel_transformation(e: Type_E) -> Type_R:
            raise NotImplementedError
        return kernel_transformation

    def define_pushforward(self, kernel_transformation: Type_E_R) -> Type_V_NR:
        def pushforward(v: Type_V) -> Type_NR:
            raise NotImplementedError
        return pushforward
    
    def define_aggregator(self, pushforward: Type_V_NR) -> Type_V_R:
        def aggregator(v: Type_V) -> Type_R:
            raise NotImplementedError
        return aggregator

    def update(self, output):
        raise NotImplementedError

    def pipeline(self, V: torch.Tensor, E: torch.Tensor, X: torch.Tensor):
        # Prepare edge indices for span diagram and the feature function f : V -> R
        self.E_indexed = self._add_edge_indices(E)
        self._set_preimages(V)
        self.X = X

        # Prepare pipeline
        pullback = self.define_pullback(self.f) # E -> R
        kernel_transformation = self.define_kernel(pullback) # E -> R
        pushforward = self.define_pushforward(kernel_transformation) # V -> N[R]
        aggregator = self.define_aggregator(pushforward) # V -> R

        # Apply the pipeline to each node in the graph
        updated_features = torch.Tensor()
        for v in V:
            updated_features = torch.hstack((updated_features,aggregator(v)))

        return self.update(updated_features.view(X.shape[0],-1))