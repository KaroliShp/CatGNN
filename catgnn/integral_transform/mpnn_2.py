from typing import Tuple

import torch
from torch import nn
from torch import Tensor


class BaseMPNNLayer_2(nn.Module):
    def __init__(self):
        super(BaseMPNNLayer_2, self).__init__()

    # Directed graph span constructions

    def s(self, E: Tensor) -> Tensor:
        """
        s: E -> V
        Get the sender of each node in set of edges E.

        Args:
            E (Tensor): set of edges (sender to receiver) of shape (2,-1)

        Returns:
            Tensor: senders of each edge
        """
        return E[0]

    def t(self, E: Tensor) -> Tensor:
        """
        t: E -> V
        Get the receiver of each node in set of edges E.

        Args:
            E (Tensor): set of edges (sender to receiver) of shape (2,-1)

        Returns:
            Tensor: receivers of each edge
        """
        return E[1]

    def t_1(self, V: Tensor) -> Tuple[Tensor, Tensor]:
        """
        t^{-1}: V -> P(E)
        Get preimages of nodes in V. This chooses from all E, because which E to use
        can be selected by pullback later.

        Args:
            V (Tensor): set of nodes of shape (-1,)

        Returns:
            Tuple[Tensor, Tensor]: the set of edges in all the preimages of V and
            the bag indices (which node's "bag"/preimage the edge belongs to).
        """
        selected_E = self.E.T[torch.isin(self.t(self.E), V)].T
        return selected_E, self.t(selected_E)

    def get_opposite_edges(self, E: Tensor, masking_required: bool = False) -> Tensor:
        """
        *: E -> E
        Get opposite edges of all edges in E as described in paper's appendix.

        Args:
            E (Tensor): set of edges (sender to receiver) of shape (2,-1)
            masking_required (bool, optional): if True, then masking may be required if the
            opposite edge does not exist in the graph (e.x. for a directed graph, a -> b may
            exist, but b -> a may not exist).
            Defaults to False.

        Returns:
            Tensor: set of opposite edges of the same shape as E
        """
        # Flip row 0 and row 1 to create all the opposite edges
        flipped_E = torch.flip(E, [0])

        if masking_required:
            # Need to check that all opposite edges exist so that f assigns correct value when
            # self.s(flipped_E) gives the sender

            # These values show that the edge at that index in flipped_E exists in E
            values, _ = torch.max(
                (E == flipped_E.T.unsqueeze(-1)).all(dim=1).int(), dim=1
            )

            # Inverse the mask (1 to 0 and 0 to 1)
            inverse_values = (values - 1) * (-1)

            # Assign -1 to edges that don't exist. TODO: to make this work in
            # user implementation, we also need to append extra row to X which
            # contains the "default" value (e.x. torch.nan for Bellman-Ford)
            return flipped_E.masked_fill(inverse_values, -1)
        else:
            # No masking is required (for example in GCN or GAT when we pullback
            # features from sender to the edge, we don't care if there is an edge
            # from receiver back to sender)
            return flipped_E

    def f(self, V: Tensor) -> Tensor:
        """
        f: V -> R
        Function for features of nodes in V

        Args:
            V (Tensor): set of nodes of shape (-1,)

        Returns:
            Tensor: features
        """
        return self.X[V]

    # Integral transform primitives (backwards)

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

    def update(self, X: Tensor, output: Tensor) -> Tensor:
        raise NotImplementedError

    def transform_backwards(self, V: Tensor, E: Tensor, X: Tensor,
        kernel_factor: bool = False, validate_input: bool = False,
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

        # Save in the object for span constructions
        self.X = X
        self.E = E

        # Prepare integral transform
        pullback = self.define_pullback(self.f)  # E -> R
        if kernel_factor:
            # TODO: if masking is required, this would hide extra edges as implementation detail.
            # This is not needed for our use cases so far.
            # self.X = torch.cat((self.X, torch.ones(1,*self.X.shape[1:])*torch.inf),dim=0)

            product_arrow = self.define_kernel_factor_1(pullback)  # (E -> R) x (E -> R)
            kernel_transformation = self.define_kernel_factor_2(product_arrow)  # E -> R
        else:
            kernel_transformation = self.define_kernel(pullback)  # E -> R
        pushforward = self.define_pushforward(kernel_transformation)  # V -> N[R]
        aggregator = self.define_aggregator(pushforward)  # V -> R

        # Apply the integral transform to each node in the graph, then finish with update step
        return self.update(X, aggregator(V))

    # Integral transform primitives (forwards)

    def pullback(self, E, f):
        raise NotImplementedError

    def kernel_factor_1(self, E, E_star, pulledback_features):
        raise NotImplementedError

    def kernel_factor_2(self, E, kernel_factor_1):
        raise NotImplementedError

    def kernel_transformation(self, E, pulledback_features):
        raise NotImplementedError

    def pushforward(self, V, edge_messages):
        raise NotImplementedError

    def aggregator(self, V, edge_messages, bag_indices):
        raise NotImplementedError

    def update(self, X, output):
        raise NotImplementedError

    def transform_forwards(self, V: Tensor, E: Tensor, X: Tensor,
        kernel_factor: bool = False, validate_input: bool = False,
    ) -> Tensor:
        """
        Integral transform implementation (forwards)

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

        # Save in the object for span constructions
        self.X = X

        # Pull node features along the span into edge features
        pulledback_features, chosen_E = self.pullback(E, self.f)

        if kernel_factor:
            # Not implemented at the moment (I don't particularly like forwards and I think
            # backwards implementation is better)
            raise NotImplementedError
        else:
            # Do the kernel transformation on pulled node features. We need chosen_E to know
            # which pulled features belong to which edges, because we may choose edges
            # arbitrarily. These edge messages are edge messages only for the pulled features
            edge_messages = self.kernel_transformation(chosen_E, pulledback_features)

        # For each receiver, go over its preimage edges and collect the edge messages into bags
        # We can select which receivers we want here? Or later for aggregator? TODO
        # Note that pushforward will get preimages for V from all edges
        # However, it should only get preimages out of edges that were selected by pullback
        self.E = chosen_E
        edge_messages, bag_indices = self.pushforward(V, edge_messages)

        # Aggregate for selected V
        aggregated_output = self.aggregator(V, edge_messages, bag_indices)

        # Update and return
        return self.update(X, aggregated_output)

    # Other utils

    def _validate_input(self, V: Tensor, E: Tensor, X: Tensor):
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

        # TODO: maybe also assert devices (easier to debug)?
        # TODO: maybe also create a new method for checking that the implementation is
        # correct (that is that the types of returned functions is correct etc.)?
