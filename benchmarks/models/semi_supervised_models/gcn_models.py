from catgnn.layers.gcn.gcn_mpnn_1 import GCNLayer_MPNN_1
from catgnn.layers.gcn.gcn_mpnn_2 import (
    GCNLayer_MPNN_2,
    GCNLayer_Factored_MPNN_2,
    GCNLayer_MPNN_2_Forwards,
)
import torch
from torch import nn
import torch_geometric


"""
General layers
"""


class GCN_1(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim=1,
        num_layers=1,
        factored=False,
        forwards=False,
    ):
        super().__init__()
        print("Standard implementation")
        self.chosen_layer = GCNLayer_MPNN_1

        self.num_layers = num_layers
        if self.num_layers == 1:
            self.layers = [self.chosen_layer(input_dim, output_dim)]
        else:
            self.layers = [self.chosen_layer(input_dim, hidden_dim)] + [
                self.chosen_layer(hidden_dim, hidden_dim)
                for _ in range(self.num_layers - 2)
            ]
            self.layers += [self.chosen_layer(hidden_dim, output_dim)]
        self.layers = nn.ModuleList(self.layers)

    def forward(self, V, E, X):
        if self.num_layers == 1:
            return nn.functional.relu(self.layers[0](V, E, X))

        H = nn.functional.relu(self.layers[0](V, E, X))
        for i in range(self.num_layers - 2):
            H = nn.functional.relu(self.layers[i + 1](V, E, H))
        return self.layers[-1](V, E, H)  # PyG uses log softmax here

    def __repr__(self):
        return self.__class__.__name__


class GCN_2(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim=1,
        num_layers=1,
        factored=False,
        forwards=False,
    ):
        super().__init__()
        assert (
            factored and forwards
        ) == False, "Factored and forwards not supported at the moment"
        if factored:
            print("Factored implementation")
            self.chosen_layer = GCNLayer_Factored_MPNN_2
        elif forwards:
            print("Forwards implemenation")
            self.chosen_layer = GCNLayer_MPNN_2_Forwards
        else:
            print("Standard implementation")
            self.chosen_layer = GCNLayer_MPNN_2

        self.num_layers = num_layers
        if self.num_layers == 1:
            self.layers = [self.chosen_layer(input_dim, output_dim)]
        else:
            self.layers = [self.chosen_layer(input_dim, hidden_dim)] + [
                self.chosen_layer(hidden_dim, hidden_dim)
                for _ in range(self.num_layers - 2)
            ]
            self.layers += [self.chosen_layer(hidden_dim, output_dim)]
        self.layers = nn.ModuleList(self.layers)

    def forward(self, V, E, X):
        if self.num_layers == 1:
            return nn.functional.relu(self.layers[0](V, E, X))

        H = nn.functional.relu(self.layers[0](V, E, X))
        for i in range(self.num_layers - 2):
            H = nn.functional.relu(self.layers[i + 1](V, E, H))
        return self.layers[-1](V, E, H)  # PyG uses log softmax here

    def __repr__(self):
        return self.__class__.__name__


class PyG_GCN(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=1, num_layers=1):
        super().__init__()

        self.num_layers = num_layers
        if self.num_layers == 1:
            self.layers = [
                torch_geometric.nn.conv.GCNConv(
                    input_dim, output_dim, cached=False, add_self_loops=True
                )
            ]
        else:
            self.layers = [
                torch_geometric.nn.conv.GCNConv(
                    input_dim, hidden_dim, cached=False, add_self_loops=True
                )
            ] + [
                torch_geometric.nn.conv.GCNConv(
                    hidden_dim, hidden_dim, cached=False, add_self_loops=True
                )
                for _ in range(self.num_layers - 2)
            ]
            self.layers += [
                torch_geometric.nn.conv.GCNConv(
                    hidden_dim, output_dim, cached=False, add_self_loops=True
                )
            ]
        self.layers = nn.ModuleList(self.layers)

    def forward(self, V, E, X):
        if self.num_layers == 1:
            return nn.functional.relu(self.layers[0](X, E))

        H = nn.functional.relu(self.layers[0](X, E))
        for i in range(self.num_layers - 2):
            H = nn.functional.relu(self.layers[i + 1](H, E))
        return self.layers[-1](H, E)  # PyG uses log softmax here

    def __repr__(self):
        return self.__class__.__name__


"""
Paper benchmarking (architecture choice from PyG)
"""


class GCN_1_Paper(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        print("Standard implementation")
        self.conv1 = GCNLayer_MPNN_1(input_dim, 16)
        self.conv2 = GCNLayer_MPNN_1(16, output_dim)

    def forward(self, V, E, X):
        H = nn.functional.relu(self.conv1(V, E, X))
        H = nn.functional.dropout(H, training=self.training)
        H = self.conv2(V, E, H)
        return H  # PyG uses log softmax here

    def __repr__(self):
        return self.__class__.__name__


class GCN_2_Paper(torch.nn.Module):
    def __init__(self, input_dim, output_dim, factored=False, forwards=False):
        super().__init__()
        assert (
            factored and forwards
        ) == False, "Factored and forwards not supported at the moment"
        if factored:
            print("Factored implementation")
            self.conv1 = GCNLayer_Factored_MPNN_2(input_dim, 16)
            self.conv2 = GCNLayer_Factored_MPNN_2(16, output_dim)
        elif forwards:
            print("Forwards implemenation")
            self.conv1 = GCNLayer_MPNN_2_Forwards(input_dim, 16)
            self.conv2 = GCNLayer_MPNN_2_Forwards(16, output_dim)
        else:
            print("Standard implementation")
            self.conv1 = GCNLayer_MPNN_2(input_dim, 16)
            self.conv2 = GCNLayer_MPNN_2(16, output_dim)

    def forward(self, V, E, X):
        H = nn.functional.relu(self.conv1(V, E, X))
        H = nn.functional.dropout(H, training=self.training)
        H = self.conv2(V, E, H)
        return H  # PyG uses log softmax here

    def __repr__(self):
        return self.__class__.__name__


class PyG_GCN_Paper(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.conv1 = torch_geometric.nn.conv.GCNConv(
            input_dim, 16, cached=False, add_self_loops=True
        )
        self.conv2 = torch_geometric.nn.conv.GCNConv(
            16, output_dim, cached=False, add_self_loops=True
        )

    def forward(self, V, E, X):
        H = nn.functional.relu(self.conv1(X, E))
        H = nn.functional.dropout(H, training=self.training)
        H = self.conv2(H, E)
        return H  # PyG uses log softmax here

    def __repr__(self):
        return self.__class__.__name__


if __name__ == "__main__":
    # Example graph above
    # V is a set of nodes - usual representation
    # V = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
    V = torch.tensor([0, 1, 2, 3], dtype=torch.int64)

    # E is a set of edges - usual sparse representation in PyG
    E = torch.tensor(
        [(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)], dtype=torch.int64
    ).T
    """
    E = torch.tensor([[0, 1, 1, 2, 2, 3],
                      [1, 0, 2, 1, 3, 1]], dtype=torch.int64)
    """

    # Feature matrix - usual representation
    X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)

    example_layer = PyG_GCN(2, 2, num_layers=4)
    print(example_layer(V, E, X))
