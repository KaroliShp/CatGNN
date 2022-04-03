from catgnn.integral_transform.mpnn_3 import BaseMPNNLayer_3
from catgnn.typing import *
import numpy as np
import torch
import torch_scatter


class GenericMPNNLayer_3_Forwards(BaseMPNNLayer_3):

    def __init__(self):
        super().__init__()
    
    def forward(self, V, E, X):
        out = self.pipeline_forwards(V, E, X, kernel_factor=False)
        return out

    def pullback(self, E, f):
        return f(self.s(E))

    def kernel_transformation(self, E, pulledback_features):
        return pulledback_features

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

    example_layer = GenericMPNNLayer_3_Forwards()
    print(example_layer(V, E, X))