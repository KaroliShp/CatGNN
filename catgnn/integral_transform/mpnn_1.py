from typing import List

import torch
from torch import nn
from torch import Tensor


class BaseMPNNLayer_1(nn.Module):
    """
    Integral transform base class for basic (naive) implementation from 
    mini-project section 4.1. Barely used for anything except GCN because it is 
    extremely slow. Provided to show incremental upgrade to implementation and
    associated gains.
    """

    def __init__(self):
        super(BaseMPNNLayer_1, self).__init__()

    # Directed graph span construction

    def _add_edge_indices(self, E: Tensor):
        # Doesn't overwrite E - this has a memory cost, maybe TODO change?
        self.E_indexed = torch.cat(
            (E, torch.arange(0, E.shape[1], device=E.device).view(1, -1))
        )

    def s(self, e: Tensor) -> Tensor:
        """
        s: E -> V
        Get the sender node of an edge e

        Args:
            e (Tensor): an edge of shape (1,3), where e[0] is the sender,
            e[1] is the receiver and e[2] is the index of the edge

        Returns:
            Tensor: sender node
        """
        return self.E_indexed[0][e[2]]

    def t(self, e: Tensor) -> Tensor:
        """
        t: E -> V
        Get the receiver node of an edge e

        Args:
            e (Tensor): an edge of shape (1,3), where e[0] is the sender,
            e[1] is the receiver and e[2] is the index of the edge

        Returns:
            Tensor: receiver node
        """
        return self.E_indexed[1][e[2]]

    def _set_preimages(self, V: Tensor):
        # This V is always the full V (not chosen), since its called from within the class
        self._preimages = []

        # Create a list of lists, alternatively could be a dict
        for _ in V:
            self._preimages.append([])

        # Fill in all preimages with receiver and edge index
        for i in range(0, self.E_indexed.shape[1]):
            self._preimages[self.E_indexed[1][i]].append(self.E_indexed[:, i])

    def t_1(self, v: Tensor) -> List[Tensor]:
        """
        t^{-1}: V -> P(E)
        Get preimages of node v.

        Args:
            v (Tensor): node

        Returns:
            List[Tensor]: list of edges belonging to the preimage of v
        """
        return self._preimages[v]

    def f(self, v: Tensor) -> Tensor:
        """
        f: V -> R
        Function for features of nodes in V

        Args:
            v (Tensor): node

        Returns:
            Tensor: features
        """
        return self.X[v]

    # Integral transform primitives (backwards)

    def define_pullback(self, f):
        def pullback(e):
            raise NotImplementedError

        return pullback

    def define_kernel(self, pullback):
        def kernel(e):
            raise NotImplementedError

        return kernel

    def define_pushforward(self, kernel):
        def pushforward(v):
            raise NotImplementedError

        return pushforward

    def define_aggregator(self, pushforward):
        def aggregator(v):
            raise NotImplementedError

        return aggregator

    def update(self, X, output):
        raise NotImplementedError

    def transform_backwards(self, V: Tensor, E: Tensor, X: Tensor,
        kernel_factor: bool = False, validate_input: bool = True,
    ) -> Tensor:
        """
        Integral transform implementation (backwards)

        Args:
            V (Tensor): set of nodes of shape (-1,) (each node from 0 to |V|-1)
            E (Tensor): set of edges (sender to receiver) of shape (2,-1)
            X (Tensor): set of features for each node in V of shape (|V|,num_features)
            kernel_factor (bool, optional): whether to factorise the kernel arrow to have a
            receiver-dependent kernel. Defaults to False.
            validate_input (bool, optional): whether to validate input before performing
            integral transform. Defaults to False.

        Returns:
            Tensor: set of updated features
        """
        if validate_input:
            self._validate_input(V, E, X)

        # Span construction
        self._add_edge_indices(E)
        self._set_preimages(V)
        self.X = X

        # Prepare integral transform
        pullback = self.define_pullback(self.f)  # E -> R
        if kernel_factor:
            # Factoring kernel for MPNN_1 is not supported (no point because it is already slow)
            raise NotImplementedError
        else:
            kernel = self.define_kernel(pullback)  # E -> R
        pushforward = self.define_pushforward(kernel)  # V -> N[R]
        aggregator = self.define_aggregator(pushforward)  # V -> R

        # Apply the integral transform to each node in the graph, then finish with update step
        updated_features = torch.Tensor().to(V.device)
        for v in V:
            updated_features = torch.hstack((updated_features, aggregator(v)))
        return self.update(X, updated_features.view(X.shape[0], -1))

    # Integral transform primitives (forwards)

    def transform_forwards(self, V: Tensor, E: Tensor, X: Tensor,
        kernel_factor: bool = False, validate_input: bool = True,
    ) -> Tensor:
        # Forwards implementaton is not supported (no point because it is already slow)
        raise NotImplementedError

    # Other utils

    def _validate_input(self, V, E, X):
        # Firstly assert input types
        assert type(V) == torch.Tensor, f"V must be a torch.Tensor, not {type(V)}"
        assert type(E) == torch.Tensor, f"E must be a torch.Tensor, not {type(E)}"
        assert type(X) == torch.Tensor, f"X must be a torch.Tensor, not {type(X)}"

        # Then assert tensor properties
        assert V.dtype == torch.int64, f"V.dtype must be torch.int64, not {V.dtype}"
        assert len(V.shape) == 1, f"V must be a 1D tensor"
        assert V.shape[0] != 0, f"V cannot be empty"

        assert E.dtype == torch.int64, f"E.dtype must be torch.int64, not {E.dtype}"
        assert len(E.shape) == 2, f"V must be a 2D tensor"
        assert E.shape[0] == 2, f"E must have two rows/must have shape of (2,-1)"
        assert E.shape[1] != 0, f"E cannot be empty"

        assert X.shape[0] != 0, f"X cannot be empty"

        # Now assert that the shapes of V, E and X together make sense
        assert V.shape[0] == X.shape[0], f"V must have the same shape as X"
