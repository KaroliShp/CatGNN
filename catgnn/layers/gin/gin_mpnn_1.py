from catgnn.integral_transform.mpnn_1 import BaseMPNNLayer_1
import torch
from torch import nn
import torch_geometric

 
class GINLayer_MPNN_1(BaseMPNNLayer_1):

    def __init__(self, mlp_update, eps: float=0.0, train_eps: bool=True):
        super().__init__()

        self.eps_0 = eps
        self.train_eps = train_eps
        self.eps = nn.Parameter(torch.Tensor([eps]), requires_grad=train_eps)
        self.mlp_update = mlp_update
    
    def forward(self, V, E, X):
        # Do integral transform
        return self.transform_backwards(V, E, X)

    def define_pullback(self, f):
        def pullback(e):
            return f(self.s(e))
        return pullback
    
    def define_kernel(self, pullback):
        def kernel_transformation(e):
            return pullback(e)
        return kernel_transformation
    
    def define_pushforward(self, kernel_transformation):
        def pushforward(v):
            pE = self.t_1(v)

            # Now we need to apply edge_messages function for each element of pE
            bag_of_messages = []
            for e in pE:
                bag_of_messages.append(kernel_transformation(e))
            return bag_of_messages
        return pushforward

    def define_aggregator(self, pushforward):
        def aggregator(v):
            # Sum aggregator
            total = 0
            for val in pushforward(v):
                total += val
            return total
        return aggregator

    def update(self, X, output):
        return self.mlp_update(((1+self.eps)*X) + output)

    def reset_parameters(self):
        torch_geometric.nn.inits.reset(self.mlp_update)
        self.eps = nn.Parameter(torch.Tensor([self.eps_0]), requires_grad=self.train_eps)


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