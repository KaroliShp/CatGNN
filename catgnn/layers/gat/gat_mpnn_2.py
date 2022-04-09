from catgnn.integral_transform.mpnn_2 import BaseMPNNLayer_2
from catgnn.typing import *
import torch
from torch import nn
import torch_scatter


class GATLayer_MPNN_2(BaseMPNNLayer_2):

    def __init__(self, in_dim: int, out_dim: int):
        # Start with only 1 attention head for simplicity
        super().__init__()

        self.mlp_msg = nn.Linear(in_dim, out_dim) # \psi

        self.attention_a = nn.Linear(in_dim*2, out_dim)

        self.mlp_update = nn.LeakyReLU(negative_slope=0.02) # \phi
    
    def forward(self, V, E, X):
        # Add self-loops to the adjacency matrix
        # GAT paper: " In all our experiments, these will be exactly the first-order neighbors of i (including i)"
        E = torch.cat((E,torch.arange(V.shape[0]).repeat(2,1)), dim=1)

        # Do integral transform
        return self.pipeline_backwards(V, E, X, kernel_factor=True)
    
    def define_pullback(self, f):
        def pullback(E):
            return f(self.s(E))
        return pullback

    def define_kernel_factor_1(self, pullback):
        def kernel_factor_1(E):
            E_star = self.get_opposite_edges(E)
            return pullback(E), pullback(E_star)
        return kernel_factor_1

    def define_kernel_factor_2(self, kernel_factor_1):
        def kernel_factor_2(E):
            # Get features of both senders and receivers
            r_sender, r_receiver = kernel_factor_1(E)

            """
            print('Shapes:')
            print(r_sender)
            print(r_sender.shape)
            print(r_receiver)
            print(r_receiver.shape)
            print('')
            """

            # Attention: calculate a_{i,j}
            concatenated_features = torch.cat((r_sender, r_receiver),-1) # Note that we concatenate on the last dimension (-1) (rows)
            #print(concatenated_features.shape)
            attention_coefficients = torch.nn.functional.leaky_relu(self.attention_a(concatenated_features)).exp() # e_{i,j}
            """
            print(attention_coefficients)
            print(attention_coefficients.shape)
            """
            softmax_denominator = torch_scatter.scatter_add(attention_coefficients, self.t(E), dim=0)  # Must be same shape as X
            """
            print(softmax_denominator)
            print(softmax_denominator.shape)
            print('\nChosen demoninators:')
            print(softmax_denominator[self.t(E)])
            print(softmax_denominator[self.t(E)].shape)
            print(5/0)
            """
            softmaxed_coefficients = attention_coefficients / softmax_denominator[self.t(E)]
            """
            print(softmaxed_coefficients.shape)
            print('')
            print(5/0)
            """

            # Perform kernel transform
            return softmaxed_coefficients * self.mlp_msg(r_sender)
        return kernel_factor_2
    
    def define_pushforward(self, kernel):
        def pushforward(V):
            E, bag_indices = self.t_1(V)
            return kernel(E), bag_indices
        return pushforward
    
    def define_aggregator(self, pushforward):
        def aggregator(V):
            edge_messages, bag_indices = pushforward(V)
            aggregated = torch_scatter.scatter_add(edge_messages.T, bag_indices.repeat(edge_messages.T.shape[0],1)).T
            return aggregated[V]
        return aggregator

    def update(self, X, output):
        return self.mlp_update(output)


if __name__ == '__main__':
    V = torch.tensor([0, 1, 2, 3], dtype=torch.int64)

    # E is a set of edges - usual sparse representation in PyG
    E = torch.tensor([[0, 1, 1, 2, 2, 3],
                      [1, 0, 2, 1, 3, 1]], dtype=torch.int64)

    # Feature matrix - usual representation
    X = torch.tensor([[0,0], [0,1], [1,0], [1,1]], dtype=torch.float)

    example_layer = GATLayer_MPNN_2(2,2)
    print(example_layer(V, E, X))