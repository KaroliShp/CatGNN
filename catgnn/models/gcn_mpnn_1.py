from catgnn.integral_transform.mpnn_1 import BaseMPNNLayer_1
from catgnn.typing import *
import torch
from torch import nn
import numpy as np

 
class GCNLayer_MPNN_1(BaseMPNNLayer_1):

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()

        self.mlp_msg = nn.Linear(in_dim, out_dim) # \psi
        self.mlp_update = nn.LeakyReLU() # \phi
    
    def forward(self, V: List[Type_V], E: List[Type_E], X: Type_R) -> Type_R:
        # 1. Add self loops to the adjacency matrix
        for v in V:
            E = np.concatenate((E, [(v,v)]), axis=0)
        
        # 3. Compute normalization and provide as edge features
        self.deg_i = {}
        for e in E:
            if e[0] in self.deg_i:
                self.deg_i[e[0]] += 1
            else:
                self.deg_i[e[0]] = 1
        self.norm = {}
        for e in E:
            assert tuple(e) not in self.norm
            self.norm[tuple(e)] = torch.sqrt(1/torch.tensor(self.deg_i[e[0]] * self.deg_i[e[1]]))

        # Do pipeline
        out = self.pipeline(V, E, X)

        # Do update (non-linearity)
        return self.mlp_update(out)

    def define_pullback(self, f: Type_V_R) -> Type_E_R:
        def pullback(e: Type_E) -> Type_R:
            return f(self.s(e))
        return pullback
    
    def define_kernel(self, er: Type_E_R) -> Type_E_R:
        def kernel_transformation(e: Type_E) -> Type_R:
            # 2. Linearly transform node feature matrix and 4. Normalize node features
            return self.mlp_msg(er(e).float()) * self.norm[tuple(e)]
        return kernel_transformation
    
    def define_pushforward(self, edge_messages: Type_E_R) -> Type_V_NR:
        def pushforward(v: Type_V) -> Type_NR:
            pE = self.t_1(v)

            # Now we need to apply edge_messages function for each element of pE
            bag_of_messages = []
            for e in pE:
                bag_of_messages.append(edge_messages(e))
            return bag_of_messages
        return pushforward

    def define_aggregator(self, bag_of_values: Type_V_NR) -> Type_V_R:
        def aggregator(v: Type_V) -> Type_R:
            total = 0
            for val in bag_of_values(v):
                total += val
            return total
        return aggregator


if __name__ == '__main__':
    # Example graph above
    # V is a set of nodes - usual representation
    V = np.array([0, 1, 2, 3])

    # E is a set of edges - usual sparse representation in PyG
    E = np.array([(0,1), (1,0),
                (1,2), (2,1),
                (2,3), (3,2)
    ])

    # Feature matrix - usual representation
    X = torch.tensor([[0,0], [0,1], [1,0], [1,1]])

    gcn_layer = GCNLayer_MPNN_1(2,2)
    #print([p for p in gcn_layer.parameters()])
    print(gcn_layer(V, E, X))