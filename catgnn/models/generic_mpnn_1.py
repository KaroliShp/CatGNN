from catgnn.integral_transform.mpnn_1 import BaseMPNNLayer_1
from catgnn.typing import *
import numpy as np
import torch


class GenericMPNNLayer(BaseMPNNLayer_1):

    def __init__(self):
        super().__init__()
    
    def forward(self, V: List[Type_V], E: List[Type_E], X: Type_R) -> Type_R:
        out = self.pipeline(V, E, X)
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

    def update(self, output):
        return output