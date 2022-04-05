import torch
from torch import nn
from catgnn.typing import *


class BaseMPNNLayer_3(nn.Module):
    def __init__(self):
        super(BaseMPNNLayer_3, self).__init__()

    """
    Directed graph span construction
    """

    def s(self, E):
        return E[1]

    def t(self, E):
        return E[0]

    def t_1(self, V):
        # Get preimages of only those edges where the receiver is in V
        # Return the edges in the preimage and the bag indices
        # (which node's "bag"/preimage the edge belongs to)
        # This chooses from all E, because which E to use can be selected by pullback later
        selected_E = self.E.T[torch.isin(self.t(self.E), V)].T
        return selected_E, self.t(selected_E)

    def t_1_chosen_E(self, V):
        # This is identical to self.t_1, however preimages are gotten based on the selected edges,
        # not all edges (because selection has already happened)
        selected_E = self.chosen_E.T[torch.isin(self.t(self.chosen_E), V)].T
        return selected_E, self.t(selected_E)
    
    def f(self, V):
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

    def pipeline_backwards(self, V, E, X, kernel_factor=False):
        # Set the span diagram (preimage) and feature function f : V -> R
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

    def pushforward(self, V, edge_messages):
        raise NotImplementedError
    
    def aggregator(self, V, edge_messages, bag_indices):
        raise NotImplementedError

    def update(self, X, output):
        raise NotImplementedError

    def pipeline_forwards(self, V, E, X, kernel_factor=False):    
        # Set the span diagram (preimage) and feature function f : V -> R
        self.X = X

        # Pull node features along the span into edge features
        pulledback_features, chosen_E = self.pullback(E, self.f)
        #print(f'pulledback features: {pulledback_features}\n')

        # Do the kernel transformation on pulled node features
        # We need selected_E to know which pulledback features belong to which edges
        # These edge messages are only edge messages for the pulled back features
        # In other words, assert edge_messages.shape[0] == pulledback_features.shape[0]
        edge_messages = self.kernel_transformation(chosen_E, pulledback_features)
        #print(f'edge messages: {edge_messages}\n')

        # For each receiver, go over their preimage edges and collect the edge messages into bags
        # We can select which receivers we want here? Or later for aggregator? TODO
        # Note that pushforward will get preimages for V from all edges
        # However, it should only get preimages out of edges that were selected by pullback
        self.chosen_E = chosen_E
        edge_messages, bag_indices = self.pushforward(V, edge_messages)
        #print(f'edge messages: {edge_messages}, bag indices: {bag_indices}\n')

        # Aggregate for selected V
        aggregated_output = self.aggregator(V, edge_messages, bag_indices)
        #print(f'aggregated output: {aggregated_output}\n')

        # Update and return
        return self.update(X, aggregated_output)