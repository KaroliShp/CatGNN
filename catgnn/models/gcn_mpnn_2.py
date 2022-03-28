from catgnn.integral_transform.mpnn_2 import BaseMPNNLayer_2
from catgnn.typing import *
import torch
from torch import nn
import numpy as np


class GCNLayer_MPNN_2(BaseMPNNLayer_2):

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()

        self.mlp_msg = nn.Linear(in_dim, out_dim) # \psi
        self.mlp_update = nn.LeakyReLU() # \phi
    
    def forward(self, V: torch.Tensor, E: torch.Tensor, X: torch.Tensor) -> torch.Tensor:        
        # 3. Compute normalization and provide as edge features
        self.degrees = torch.zeros(V.shape[0], dtype=E.dtype).scatter_add_(0, E.T[0], torch.ones(E.T[0].shape, dtype=E.dtype))

        self.v_counter = 0

        # Do pipeline
        out = self.pipeline(V, E, X)

        # Do update (non-linearity)
        return self.mlp_update(out)

    def define_pullback(self, f: Type_V_R) -> Type_E_R:
        def pullback(E: torch.Tensor) -> torch.Tensor:
            return f(self.s(E))
        
        return pullback
    
    def define_kernel(self, pullback: Type_E_R) -> Type_E_R:
        def kernel_transformation(E: torch.Tensor) -> torch.Tensor:
            # 2. Linearly transform node feature matrix and 4. Normalize node features
            E = torch.cat((torch.tensor([[self.v_counter], [self.v_counter]], dtype=E.dtype), E), dim=1)
            norm = torch.sqrt(1/( (self.degrees[self.v_counter].repeat(E.shape[1], 1)+1) * (self.degrees[E[1]].view(-1,1)+1) ))
            self.v_counter += 1
            return self.mlp_msg(pullback(E)) * norm
        
        return kernel_transformation
    
    def define_pushforward(self, kernel_transformation: Type_E_R) -> Type_V_NR:
        def pushforward(V: torch.Tensor) -> torch.Tensor:
            pE = self.t_1(V)
            bag = []
            for e in pE:
                bag.append(kernel_transformation(e))
            return bag
        
        return pushforward

    def define_aggregator(self, pushforward: Type_V_NR) -> Type_V_R:
        def aggregator(V: torch.Tensor) -> torch.Tensor:
            bag = pushforward(V)
            total = torch.Tensor()
            for b in bag:
                total = torch.hstack((total, torch.sum(b, dim=0)))
            #print(total.shape)
            #print(V.shape)
            return total.reshape(V.shape[0],-1)
        
        return aggregator