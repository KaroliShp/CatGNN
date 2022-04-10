from catgnn.integral_transform.mpnn_2 import BaseMPNNLayer_2
import torch
from torch import nn
import torch_scatter
from catgnn.utils import add_self_loops, get_degrees


class GCNLayer_MPNN_2(BaseMPNNLayer_2):

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()

        self.mlp_msg = nn.Linear(in_dim, out_dim) # \psi
        self.mlp_update = nn.LeakyReLU() # \phi
    
    def forward(self, V, E, X):
        # Add self-loops to the adjacency matrix.
        #E = torch.cat((E,torch.arange(V.shape[0]).repeat(2,1)), dim=1)
        E = add_self_loops(V, E)

        # Compute normalization.
        #self.degrees = torch.zeros(V.shape[0], dtype=torch.int64).scatter_add_(0, E[1], torch.ones(E.shape[1], dtype=torch.int64))
        self.degrees = get_degrees(V, E)
        self.norm = torch.sqrt(1/(self.degrees[E[0]] * self.degrees[E[1]]))

        # Do integral transform
        return self.pipeline_backwards(V, E, X)

    def define_pullback(self, f):
        def pullback(E):
            return f(self.s(E))
        return pullback

    def define_kernel(self, pullback):
        def kernel(E):
            return self.mlp_msg(pullback(E)) * self.norm.view(-1, 1)
        return kernel

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
        return output # dont forget to fix it back after benchmark tests are rewritten

    """
    Other methods (TODO)
    """

    def reset_parameters(self):
        self.mlp_msg.reset_parameters()


class GCNLayer_Factored_MPNN_2(BaseMPNNLayer_2):

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()

        self.mlp_msg = nn.Linear(in_dim, out_dim) # \psi
        self.mlp_update = nn.LeakyReLU() # \phi
    
    def forward(self, V, E, X):
        # Add self-loops to the adjacency matrix.
        E = torch.cat((E,torch.arange(V.shape[0]).repeat(2,1)), dim=1)

        # Compute normalization.
        self.degrees = torch.zeros(V.shape[0], dtype=torch.int64).scatter_add_(0, E[1], torch.ones(E.shape[1], dtype=torch.int64))
        self.norm = torch.sqrt(1/(self.degrees[E[0]] * self.degrees[E[1]]))

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
            r_sender, r_receiver = kernel_factor_1(E)
            return self.mlp_msg(r_sender) * self.norm.view(-1, 1)
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


class GCNLayer_MPNN_2_Forwards(BaseMPNNLayer_2):

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()

        self.mlp_msg = nn.Linear(in_dim, out_dim) # \psi
        self.mlp_update = nn.LeakyReLU() # \phi
    
    def forward(self, V, E, X):
        # Add self-loops to the adjacency matrix.
        E = torch.cat((E,torch.arange(V.shape[0]).repeat(2,1)), dim=1)

        # Compute normalization.
        self.degrees = torch.zeros(V.shape[0], dtype=torch.int64).scatter_add_(0, E[1], torch.ones(E.shape[1], dtype=torch.int64))
        self.norm = torch.sqrt(1/(self.degrees[E[0]] * self.degrees[E[1]]))

        # Do integral transform
        return self.pipeline_forwards(V, E, X)

    def pullback(self, E, f):
        return f(self.s(E)), E

    def kernel_transformation(self, E, pulledback_features):
        return self.mlp_msg(pulledback_features) * self.norm.view(-1, 1)

    def pushforward(self, V, edge_messages):
        E, bag_indices = self.t_1(V)
        return edge_messages, bag_indices
    
    def aggregator(self, V, edge_messages, bag_indices):
        aggregated = torch_scatter.scatter_add(edge_messages.T, 
                                               bag_indices.repeat(edge_messages.T.shape[0],1)).T
        return aggregated[V]

    def update(self, X, output):
        return self.mlp_update(output)