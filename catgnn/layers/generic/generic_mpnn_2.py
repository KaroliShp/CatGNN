import torch
import torch_scatter

from catgnn.integral_transform.mpnn_2 import BaseMPNNLayer_2


class GenericMPNNLayer_2(BaseMPNNLayer_2):
    """
    Generic MPNN layer using standard (backwards) implementation with BaseMPNNLayer_2.
    Kernel simply leaves the pulled node features unchanged and propagates them further.
    Used for basic testing purposes.
    """

    def __init__(self):
        super().__init__()

    def forward(self, V, E, X):
        out = self.transform_backwards(V, E, X, kernel_factor=False)
        return out

    def define_pullback(self, f):
        def pullback(E):
            # Suppose you only choose some edges here and return only results for those edges
            return f(self.s(E))

        return pullback

    def define_kernel(self, pullback):
        def kernel(E):
            # Pullback will choose edges from E, so we only need to define kernel for general E here
            return pullback(E)

        return kernel

    def define_pushforward(self, kernel):
        def pushforward(V):
            # Now we have V. We need to get preimages pE for each v in V.
            # At this point, we can choose which V to update / collect neighbours information.
            # However, we can assume we have all E, because at this point we dont choose 
            # which E to use for the GNN (this will be chosen by kernel - pullback).
            # Only those edges chosen by pullback and transformed by kernel will be returned, 
            # so again we don't need to do anything else with edges here
            E, bag_indices = self.t_1(V)

            # Problem here is that indices remain the same from all nodes and their preimages, 
            # while kernel may choose some E, so we need to discard unnecessary bag_indices at some point.
            # So if GNN has to choose its own edges, *user* needs to account for that in return types, e.x.
            # pullback needs to return both f(self.s(E)) AND filtered bag_indices for only chosen edges,
            # similarly for kernel transformation.
            return kernel(E), bag_indices

        return pushforward

    def define_aggregator(self, pushforward):
        def aggregator(V):
            edge_messages, bag_indices = pushforward(V)
            aggregated = torch_scatter.scatter_add(
                edge_messages.T, bag_indices.repeat(edge_messages.T.shape[0], 1)
            ).T
            return aggregated[V]

        return aggregator

    def update(self, X, output):
        # This does not by itself account for the fact that output can have a different shape to X.
        # For example if only some of the nodes are chosen in aggregator to be updated.
        # User has to take care of this
        return output


class GenericFactoredMPNNLayer_2(BaseMPNNLayer_2):
    """
    Generic MPNN layer using standard (backwards) implementation with BaseMPNNLayer_2 and
    factorization of kernel arrow for testing purposes. Kernel simply adds up sender and
    receiver node features. Used for basic testing purposes.
    """

    def __init__(self):
        super().__init__()

    def forward(self, V, E, X):
        out = self.transform_backwards(V, E, X, kernel_factor=True)
        return out

    def define_pullback(self, f):
        def pullback(E):
            return f(self.s(E))

        return pullback

    def define_kernel_factor_1(self, pullback):
        def kernel_factor_1(E):
            E_star = self.get_opposite_edges(E)
            return pullback(E), pullback(E_star)

        return kernel_factor_1

    def define_kernel_factor_2(self, kernel_factor_1):
        def kernel_factor_2(E):
            r_sender, r_receiver = kernel_factor_1(E)
            # Add features for testing purposes
            return r_sender + r_receiver

        return kernel_factor_2

    def define_pushforward(self, kernel):
        def pushforward(V):
            E, bag_indices = self.t_1(V)
            return kernel(E), bag_indices

        return pushforward

    def define_aggregator(self, pushforward):
        def aggregator(V):
            edge_messages, bag_indices = pushforward(V)
            aggregated = torch_scatter.scatter_add(
                edge_messages.T, bag_indices.repeat(edge_messages.T.shape[0], 1)
            ).T
            return aggregated[V]

        return aggregator

    def update(self, X, output):
        return output


class GenericMPNNLayer_2_Forwards(BaseMPNNLayer_2):
    """
    Generic MPNN layer using forwards implementation with BaseMPNNLayer_2.
    Used for basic testing purposes.
    """

    def __init__(self):
        super().__init__()

    def forward(self, V, E, X):
        out = self.transform_forwards(V, E, X, kernel_factor=False)
        return out

    def pullback(self, E, f):
        return f(self.s(E)), E

    def kernel(self, E, pulledback_features):
        return pulledback_features

    def pushforward(self, V, edge_messages):
        E, bag_indices = self.t_1(V)  # Here we don't really need E?
        return edge_messages, bag_indices

    def aggregator(self, V, edge_messages, bag_indices):
        aggregated = torch_scatter.scatter_add(
            edge_messages.T, bag_indices.repeat(edge_messages.T.shape[0], 1)
        ).T
        return aggregated[V]

    def update(self, X, output):
        return output


class GenericFactoredMPNNLayer_2_Forwards(BaseMPNNLayer_2):
    """
    Generic MPNN layer using forwards implementation with BaseMPNNLayer_2 and factorization
    of kernel arrow. Used for basic testing purposes.

    NOT IMPLEMENTED
    """

    def __init__(self):
        super().__init__()

    def forward(self, V, E, X):
        out = self.transform_forwards(V, E, X, kernel_factor=False)
        return out

    def pullback(self, E, f):
        return f(self.s(E)), E

    def kernel_factor_1(self, E, E_star, pulledback_features):
        raise NotImplementedError

    def kernel_factor_2(self, E, kernel_factor_1):
        raise NotImplementedError

    def pushforward(self, V, edge_messages):
        E, bag_indices = self.t_1_chosen_E(V)  # Here we don't really need E?
        return edge_messages, bag_indices

    def aggregator(self, V, edge_messages, bag_indices):
        aggregated = torch_scatter.scatter_add(
            edge_messages.T, bag_indices.repeat(edge_messages.T.shape[0], 1)
        ).T
        return aggregated[V]

    def update(self, X, output):
        return output