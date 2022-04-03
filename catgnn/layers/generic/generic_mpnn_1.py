from catgnn.integral_transform.mpnn_1 import BaseMPNNLayer_1
from catgnn.typing import *
import numpy as np
import torch


class GenericMPNNLayer_1(BaseMPNNLayer_1):

    def __init__(self):
        super().__init__()
    
    def forward(self, V, E, X):
        out = self.pipeline_backwards(V, E, X)
        return out

    def define_pullback(self, f: Type_V_R) -> Type_E_R:
        def pullback(e: Type_E) -> Type_R:
            return f(self.s(e))
        
        return pullback
    
    def define_kernel(self, pullback: Type_E_R) -> Type_E_R:
        def kernel_transformation(e: Type_E) -> Type_R:
            return pullback(e)
        
        return kernel_transformation
    
    def define_pushforward(self, kernel_transformation: Type_E_R) -> Type_V_NR:
        def pushforward(v: Type_V) -> Type_NR:
            pE = self.t_1(v)

            # Now we need to apply edge_messages function for each element of pE
            bag_of_messages = []
            for e in pE:
                bag_of_messages.append(kernel_transformation(e))
            return bag_of_messages
        
        return pushforward

    def define_aggregator(self, pushforward: Type_V_NR) -> Type_V_R:
        def aggregator(v: Type_V) -> Type_R:
            total = 0
            for val in pushforward(v):
                total += val
            return total
        
        return aggregator

    def update(self, X, output):
        return output


class GenericFactoredMPNNLayer_1(BaseMPNNLayer_1):

    def __init__(self):
        super().__init__()
    
    def forward(self, V: List[Type_V], E: List[Type_E], X: Type_R) -> Type_R:
        out = self.pipeline_backwards(V, E, X, kernel_factor=True)
        return out

    def define_pullback(self, f: Type_V_R) -> Type_E_R:
        def pullback(e: Type_E) -> Type_R:
            return f(self.s(e))
        
        return pullback
    
    def define_kernel_factor_1(self, pullback):
        """
        Inputs:
        E -> R

        Outputs:
        (E,E) -> (R,R)
        """
        def kernel_factor_1(e, e_star):
            """
            Product arrow
            Inputs:
            (E, E)
            
            Outputs:
            (R, R)
            """
            return pullback(e), pullback(e_star)
        return kernel_factor_1

    def define_kernel_factor_2(self, kernel_factor_1):
        """
        Inputs:
        (E,E) -> (R,R)

        Outputs:
        E -> R
        """
        def kernel_factor_2(e):
            """
            Inputs:
            E
            
            Outputs:
            R
            """
            r, r_star = kernel_factor_1(e, self.get_opposite_edge(e))
            # Ignore receiver features (r_star)
            return r
        return kernel_factor_2
    
    def define_pushforward(self, kernel_transformation: Type_E_R) -> Type_V_NR:
        def pushforward(v: Type_V) -> Type_NR:
            pE = self.t_1(v)

            # Now we need to apply edge_messages function for each element of pE
            bag_of_messages = []
            for e in pE:
                bag_of_messages.append(kernel_transformation(e))
            return bag_of_messages
        
        return pushforward

    def define_aggregator(self, pushforward: Type_V_NR) -> Type_V_R:
        def aggregator(v: Type_V) -> Type_R:
            total = 0
            for val in pushforward(v):
                total += val
            return total
        
        return aggregator

    def update(self, X, output):
        return output


"""
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

    example_layer = GenericFactoredMPNNLayer()
    print(example_layer(V, E, X))
"""