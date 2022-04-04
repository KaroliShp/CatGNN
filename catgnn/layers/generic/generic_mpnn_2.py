from catgnn.integral_transform.mpnn_2 import BaseMPNNLayer_2
from catgnn.typing import *
import numpy as np
import torch


class GenericMPNNLayer_2(BaseMPNNLayer_2):

    def __init__(self):
        super().__init__()
    
    def forward(self, V, E, X):
        out = self.pipeline_backwards(V, E, X)
        return out

    def define_pullback(self, f):
        def pullback(E):
            return f(self.s(E))
        return pullback
    
    def define_kernel(self, pullback):
        def kernel_transformation(E):
            return pullback(E)
        return kernel_transformation
    
    def define_pushforward(self, kernel_transformation):
        def pushforward(V):
            pE = self.t_1(V)
            #print(f'pE: {pE}')
            bag = []
            # This is not really e anymore, but E (tensor of edges coming from a node)
            for e in pE:
                #print(f'e: {e}')
                bag.append(kernel_transformation(e))
            return bag
        
        return pushforward

    def define_aggregator(self, pushforward):
        def aggregator(V):
            bag = pushforward(V)
            total = torch.Tensor()
            #print(f'BAG: {bag}')
            for b in bag:
                total = torch.hstack((total, torch.sum(b, dim=0)))
            return total.reshape(-1,2)
        
        return aggregator

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

    example_layer = GenericMPNNLayer_2()
    print(example_layer(V, E, X))