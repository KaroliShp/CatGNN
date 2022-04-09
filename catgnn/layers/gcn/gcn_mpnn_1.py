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
        # Add self-loops to the adjacency matrix.
        E = torch.cat((E,torch.arange(V.shape[0]).repeat(2,1)), dim=1)

        # Compute normalization.
        self.degrees = torch.zeros(V.shape[0], dtype=torch.int64).scatter_add_(0, E[1], torch.ones(E.shape[1], dtype=torch.int64))
        self.norm = torch.sqrt(1/(self.degrees[E[0]] * self.degrees[E[1]]))

        # Do integral transform
        return self.pipeline_backwards(V, E, X)

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

    def update(self, X, output):
        return self.mlp_update(output)


class GCNLayer_Factored_MPNN_1(BaseMPNNLayer_1):

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()

        self.mlp_msg = nn.Linear(in_dim, out_dim) # \psi
        self.mlp_update = nn.LeakyReLU() # \phi
    
    def forward(self, V: torch.Tensor, E: torch.Tensor, X: torch.Tensor) -> torch.Tensor:        
        # Add self-loops to the adjacency matrix.
        E = torch.cat((E,torch.arange(V.shape[0]).repeat(2,1)), dim=1)

        # Compute normalization.
        self.degrees = torch.zeros(V.shape[0], dtype=torch.int64).scatter_add_(0, E[1], torch.ones(E.shape[1], dtype=torch.int64))
        self.norm = torch.sqrt(1/(self.degrees[E[0]] * self.degrees[E[1]]))
        
        # Do integral transform
        return self.pipeline_backwards(V, E, X, kernel_factor=True)

    def define_pullback(self, f: Type_V_R) -> Type_E_R:
        def pullback(e: Type_E) -> Type_R:
            return f(self.s(e))
        return pullback

    def define_kernel_factor_1(self, pullback):
        def kernel_factor_1(e, e_star):
            return pullback(e), pullback(e_star)
        return kernel_factor_1

    def define_kernel_factor_2(self, kernel_factor_1):
        def kernel_factor_2(e):
            r, r_star = kernel_factor_1(e, self.get_opposite_edge(e))
            return self.mlp_msg(r) * self.norm[e[2]]
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
        return self.mlp_update(output)