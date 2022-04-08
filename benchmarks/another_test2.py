import torch
import torch.nn.functional as F
from benchmarks.another_test import get_planetoid_dataset, run

from torch_geometric.nn import GCNConv
from catgnn.layers.gcn.gcn_mpnn_2 import GCNLayer_MPNN_2


class Net(torch.nn.Module):
    def __init__(self, dataset):
        super().__init__()
        """
        self.conv1 = GCNConv(dataset.num_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)
        """
        self.conv1 = GCNLayer_MPNN_2(dataset.num_features, 16)
        self.conv2 = GCNLayer_MPNN_2(16, dataset.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        """
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
        """
        x, edge_index = data.x, data.edge_index
        v = torch.arange(0, x.shape[0], dtype=torch.int64)
        x = F.relu(self.conv1(v, edge_index, x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(v, edge_index, x)
        return F.log_softmax(x, dim=1)


dataset = get_planetoid_dataset('Cora', True)
run(dataset, Net(dataset), 5, 200, 0.01, 0.0005, 10)