from catgnn.integral_transform.mpnn_1 import BaseMPNNLayer_1
from catgnn.typing import *
import torch
from torch import nn

 
class GCNLayer_MPNN_1(BaseMPNNLayer_1):

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()

        self.mlp_msg = nn.Linear(in_dim, out_dim) # \psi
        self.mlp_update = nn.LeakyReLU() # \phi
    
    def forward(self, V: torch.Tensor, E: torch.Tensor, X: torch.Tensor) -> torch.Tensor:        
        # 3. Compute degrees for normalization
        self.degrees = torch.zeros(V.shape[0], dtype=E.dtype).scatter_add_(0, E.T[0], torch.ones(E.T[0].shape, dtype=E.dtype)) + 1

        # 1. Add self loops to the adjacency matrix
        E = torch.cat((E,torch.arange(V.shape[0]).repeat(2,1)), dim=1)

        # 3. Compute normalization and provide as edge features for kernel transform
        self.norm = torch.sqrt(1/self.degrees[E[0]] * self.degrees[E[1]])

        # Do integral transform
        return self.pipeline(V, E, X)

    def define_pullback(self, f: Type_V_R) -> Type_E_R:
        def pullback(e: Type_E) -> Type_R:
            return f(self.s(e))
        return pullback
    
    def define_kernel(self, pullback: Type_E_R) -> Type_E_R:
        def kernel_transformation(e: Type_E) -> Type_R:
            # 2. Linearly transform node feature matrix and 4. Normalize node features
            return self.mlp_msg(pullback(e)) * self.norm[e[2]]
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
        return self.mlp_update(output)