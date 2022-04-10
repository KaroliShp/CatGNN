import torch
from torch import nn


class BaseMPNNLayer_1(nn.Module):

    def __init__(self):
        super(BaseMPNNLayer_1, self).__init__()

    """
    Directed graph span construction
    """

    def _add_edge_indices(self, E):
        self.E_indexed = torch.cat((E, torch.arange(0, E.shape[1]).view(1,-1)))

    def s(self, e):
        return self.E_indexed[0][e[2]]

    def t(self, e):
        return self.E_indexed[1][e[2]]

    def _set_preimages(self, V):
        # This V always has to be full V (not chosen), since it is passed to self.pipeline()
        self._preimages = []

        # Create a list of lists, alternatively could be a dict
        for _ in V:
            self._preimages.append([])

        # Fill in all preimages with receiver and edge index
        for i in range(0, self.E_indexed.shape[1]):
            self._preimages[self.E_indexed[1][i]].append(self.E_indexed[:,i])

    def t_1(self, v):
        return self._preimages[v]

    def f(self, v):
        return self.X[v]

    """
    Integral transform primitives (backwards)
    """

    def define_pullback(self, f):
        def pullback(e):
            raise NotImplementedError
        return pullback

    def define_kernel(self, pullback):
        def kernel_transformation(e):
            raise NotImplementedError
        return kernel_transformation

    def define_pushforward(self, kernel_transformation):
        def pushforward(v):
            raise NotImplementedError
        return pushforward
    
    def define_aggregator(self, pushforward):
        def aggregator(v):
            raise NotImplementedError
        return aggregator

    def update(self, X, output):
        raise NotImplementedError

    def pipeline_backwards(self, V, E, X, kernel_factor=False):
        # Prepare edge indices for span diagram and the feature function f : V -> R
        self._add_edge_indices(E)
        self._set_preimages(V)
        self.X = X

        # Prepare pipeline
        pullback = self.define_pullback(self.f) # E -> R
        if kernel_factor:
            # Factoring kernel for MPNN_1 is not supported (no point because it is already slow)
            raise NotImplementedError
        else:
            kernel_transformation = self.define_kernel(pullback) # E -> R
        pushforward = self.define_pushforward(kernel_transformation) # V -> N[R]
        aggregator = self.define_aggregator(pushforward) # V -> R

        # Apply the pipeline to each node in the graph
        updated_features = torch.Tensor()
        for v in V:
            updated_features = torch.hstack((updated_features,aggregator(v)))

        return self.update(X, updated_features.view(X.shape[0],-1))

    """
    Integral transform primitives (backwards)
    """

    def pipeline_forwards(self, V, E, X, kernel_factor=False):
        # Forwards implementaton is not supported (no point because it is already slow)
        raise NotImplementedError