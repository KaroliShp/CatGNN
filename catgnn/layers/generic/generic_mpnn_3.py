from catgnn.integral_transform.mpnn_3 import BaseMPNNLayer_3
from catgnn.typing import *
import numpy as np
import torch
import torch_scatter


class GenericMPNNLayer_3(BaseMPNNLayer_3):

    def __init__(self):
        super().__init__()
    
    def forward(self, V, E, X):
        out = self.pipeline_backwards(V, E, X, kernel_factor=False)
        return out
    
    def define_pullback(self, f):
        def pullback(E):
            # Suppose you only choose some edges here and return only results for those edges
            return f(self.s(E))
        return pullback

    def define_kernel(self, pullback):
        def kernel(E):
            # Pullback will choose which edges in E are worth keeping, so we only need to define kernel for those edges
            # Like obv you don't perform transformation on edges that don't exist
            return pullback(E)
        return kernel
    
    def define_pushforward(self, kernel):
        def pushforward(V):
            # Now we have V. We need to get preimages pE for each v in V
            # Can we assume this is all V? - Probably not. At this point this basically amounts to choosing V
            # However, we can assume we have all E, because at this point we dont choose which E to use for the GNN (this will be chosen by kernel - pullback)

            # Then for each E in pE we need to call kernel, this will later choose which E will be called
            # Only those edges chosen by pullback and transformed by kernel will be returned, so again we don't need to do anything else with edges here
            E, indices = self.t_1(V)

            # Problem here is that indices remain the same, while kernel may choose some E
            # Indices here indicate which bag the edge feature belongs to
            return kernel(E), indices
        return pushforward
    
    def define_aggregator(self, pushforward):
        def aggregator(V):
            edge_messages, indices = pushforward(V)
            aggregated = torch_scatter.scatter_add(edge_messages.T, indices.repeat(edge_messages.T.shape[0],1)).T
            return aggregated[V]
        return aggregator

    def update(self, X, output):
        return output


class GenericMPNNLayer_3_Forwards(BaseMPNNLayer_3):

    def __init__(self):
        super().__init__()
    
    def forward(self, V, E, X):
        out = self.pipeline_forwards(V, E, X, kernel_factor=False)
        return out

    def pullback(self, E, f):
        return f(self.s(E))

    def kernel_transformation(self, E, pulledback_features):
        return pulledback_features # Only pullback those that are in E?

    def pushforward(self, V, E, edge_messages):
        return edge_messages, self.t(E)
    
    def aggregator(self, V, bags_of_values):
        aggregated = torch_scatter.scatter_add(bags_of_values[0].T, 
                                               bags_of_values[1].repeat(bags_of_values[0].T.shape[0],1)).T
        return aggregated[V]

    def update(self, X, output):
        return output


if __name__ == '__main__':
    # Example graph above
    # V is a set of nodes - usual representation
    V = torch.tensor([0, 1, 2, 3], dtype=torch.int64)

    # E is a set of edges - usual sparse representation in PyG
    E = torch.tensor([(0,1), (1,0),
                      (1,2), (2,1), 
                      (2,3), (3,2) ], dtype=torch.int64).T

    # Feature matrix - usual representation
    X = torch.tensor([[0,0], [0,1], [1,0], [1,1]])

    example_layer = GenericMPNNLayer_3()
    print(example_layer(V, E, X))