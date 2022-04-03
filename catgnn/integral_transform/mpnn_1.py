import torch
from torch import nn
from catgnn.typing import *


class BaseMPNNLayer_1(nn.Module):

    def __init__(self):
        super(BaseMPNNLayer_1, self).__init__()

    """
    Directed graph span construction
    """

    def _add_edge_indices(self, E: torch.Tensor):
        self.E_indexed = torch.cat((E, torch.arange(0, E.shape[1]).view(1,-1)))

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

    def _add_opposite_edges(self):
        self.E_indexed = torch.cat((self.E_indexed, (torch.zeros(self.E_indexed.shape[1], dtype=torch.int64).view(1,-1))-1))

        for i in range(self.E_indexed.shape[1]):
            for j in range(self.E_indexed.shape[1]):
                if self.s(self.E_indexed[:,i]) == self.t(self.E_indexed[:,j]) and self.s(self.E_indexed[:,j]) == self.t(self.E_indexed[:,i]):
                    self.E_indexed[3][i] = self.E_indexed[:,j][2]

    def get_opposite_edge(self, e):
        return self.E_indexed[:,e[3]]

    def f(self, v: torch.Tensor) -> torch.Tensor:
        return self.X[v]


    """
    Integral transform primitives
    """

    def define_pullback(self, f: Type_V_R) -> Type_E_R:
        def pullback(e: Type_E) -> Type_R:
            raise NotImplementedError
        return pullback

    def define_kernel_factor_1(self, pullback):
        def kernel_factor_1(e, e_star):
            raise NotImplementedError
        return kernel_factor_1

    def define_kernel_factor_2(self, kernel_factor_1):
        def kernel_factor_2(e):
            raise NotImplementedError
        return kernel_factor_2

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

    def pipeline(self, V: torch.Tensor, E: torch.Tensor, X: torch.Tensor, kernel_factor=False):
        # Prepare edge indices for span diagram and the feature function f : V -> R
        self._add_edge_indices(E)

        # If kernel transformation is factorized, need to find opposite edges
        if kernel_factor:
            self._add_opposite_edges()

        self._set_preimages(V)
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
        updated_features = torch.Tensor()
        for v in V:
            updated_features = torch.hstack((updated_features,aggregator(v)))

        return self.update(updated_features.view(X.shape[0],-1))