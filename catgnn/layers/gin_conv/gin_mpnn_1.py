from catgnn.integral_transform.mpnn_1 import BaseMPNNLayer_1
from catgnn.typing import *
import torch
from torch import nn

 
class GINLayer_MPNN_1(BaseMPNNLayer_1):

    def __init__(self, mlp_update, eps: float=0.0):
        super().__init__()

        self.eps = nn.Parameter(torch.Tensor([eps]), requires_grad=True)
        self.mlp_update = mlp_update
    
    def forward(self, V: torch.Tensor, E: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        # Do integral transform
        return self.pipeline_backwards(V, E, X)

    def define_pullback(self, f: Type_V_R) -> Type_E_R:
        def pullback(e: Type_E) -> Type_R:
            return f(self.s(e))
        return pullback
    
    def define_kernel(self, pullback: Type_E_R) -> Type_E_R:
        def kernel_transformation(e: Type_E) -> Type_R:
            # 2. Linearly transform node feature matrix and 4. Normalize node features
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
        return self.mlp_update(((1+self.eps)*X) + output)


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

    example_layer = GINLayer_MPNN_1(X.shape[-1], 7, 1)
    print(example_layer(V, E, X))