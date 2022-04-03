from benchmarks.utils.train_supervised import train_eval_loop
from catgnn.layers.gin_conv.gin_mpnn_1 import GINLayer_MPNN_1
from catgnn.layers.gin_conv.gin_mpnn_2 import GINLayer_MPNN_2
from catgnn.layers.gin_conv.gin_mpnn_3 import GINLayer_MPNN_3
import torch
from torch import nn
from catgnn.datasets.planetoid import PlanetoidDataset
import torch_geometric
from torch_geometric.datasets import TUDataset # TODO - delete
from torch_geometric.loader import DataLoader # TODO - delete


"""
This architecture for networks is taken from PyG examples
"""


class CatGNN_GIN_1(nn.Module):
    def __init__(self, in_channels, dim, out_channels):
        super().__init__()

        self.conv1 = GINLayer_MPNN_1(
            nn.Sequential(nn.Linear(in_channels, dim), nn.BatchNorm1d(dim), nn.ReLU(),
                       nn.Linear(dim, dim), nn.ReLU()))

        self.conv2 = GINLayer_MPNN_1(
            nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(),
                       nn.Linear(dim, dim), nn.ReLU()))

        self.conv3 = GINLayer_MPNN_1(
            nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(),
                       nn.Linear(dim, dim), nn.ReLU()))

        self.conv4 = GINLayer_MPNN_1(
            nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(),
                       nn.Linear(dim, dim), nn.ReLU()))

        self.conv5 = GINLayer_MPNN_1(
            nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(),
                       nn.Linear(dim, dim), nn.ReLU()))

        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, out_channels)

    def forward(self, x, edge_index, batch):
        V = torch.arange(0,x.shape[0])
        x = self.conv1(V, edge_index, x)
        x = self.conv2(V, edge_index, x)
        x = self.conv3(V, edge_index, x)
        x = self.conv4(V, edge_index, x)
        x = self.conv5(V, edge_index, x)
        x = torch_geometric.nn.global_add_pool(x, batch)
        x = self.lin1(x).relu()
        x = torch.nn.functional.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return torch.nn.functional.log_softmax(x, dim=-1)


class CatGNN_GIN_2(nn.Module):
    def __init__(self, in_channels, dim, out_channels):
        super().__init__()

        self.conv1 = GINLayer_MPNN_2(
            nn.Sequential(nn.Linear(in_channels, dim), nn.BatchNorm1d(dim), nn.ReLU(),
                       nn.Linear(dim, dim), nn.ReLU()))

        self.conv2 = GINLayer_MPNN_2(
            nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(),
                       nn.Linear(dim, dim), nn.ReLU()))

        self.conv3 = GINLayer_MPNN_2(
            nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(),
                       nn.Linear(dim, dim), nn.ReLU()))

        self.conv4 = GINLayer_MPNN_2(
            nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(),
                       nn.Linear(dim, dim), nn.ReLU()))

        self.conv5 = GINLayer_MPNN_2(
            nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(),
                       nn.Linear(dim, dim), nn.ReLU()))

        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, out_channels)

    def forward(self, x, edge_index, batch):
        V = torch.arange(0,x.shape[0])
        x = self.conv1(V, edge_index, x)
        x = self.conv2(V, edge_index, x)
        x = self.conv3(V, edge_index, x)
        x = self.conv4(V, edge_index, x)
        x = self.conv5(V, edge_index, x)
        x = torch_geometric.nn.global_add_pool(x, batch)
        x = self.lin1(x).relu()
        x = torch.nn.functional.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return torch.nn.functional.log_softmax(x, dim=-1)


class CatGNN_GIN_3(nn.Module):
    def __init__(self, in_channels, dim, out_channels):
        super().__init__()

        self.conv1 = GINLayer_MPNN_3(
            nn.Sequential(nn.Linear(in_channels, dim), nn.BatchNorm1d(dim), nn.ReLU(),
                       nn.Linear(dim, dim), nn.ReLU()))

        self.conv2 = GINLayer_MPNN_3(
            nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(),
                       nn.Linear(dim, dim), nn.ReLU()))

        self.conv3 = GINLayer_MPNN_3(
            nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(),
                       nn.Linear(dim, dim), nn.ReLU()))

        self.conv4 = GINLayer_MPNN_3(
            nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(),
                       nn.Linear(dim, dim), nn.ReLU()))

        self.conv5 = GINLayer_MPNN_3(
            nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(),
                       nn.Linear(dim, dim), nn.ReLU()))

        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, out_channels)

    def forward(self, x, edge_index, batch):
        V = torch.arange(0,x.shape[0])
        x = self.conv1(V, edge_index, x)
        x = self.conv2(V, edge_index, x)
        x = self.conv3(V, edge_index, x)
        x = self.conv4(V, edge_index, x)
        x = self.conv5(V, edge_index, x)
        x = torch_geometric.nn.global_add_pool(x, batch)
        x = self.lin1(x).relu()
        x = torch.nn.functional.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return torch.nn.functional.log_softmax(x, dim=-1)


class PyG_GIN(nn.Module):
    def __init__(self, in_channels, dim, out_channels):
        super().__init__()

        self.conv1 = torch_geometric.nn.GINConv(
            nn.Sequential(nn.Linear(in_channels, dim), nn.BatchNorm1d(dim), nn.ReLU(),
                       nn.Linear(dim, dim), nn.ReLU()))

        self.conv2 = torch_geometric.nn.GINConv(
            nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(),
                       nn.Linear(dim, dim), nn.ReLU()))

        self.conv3 = torch_geometric.nn.GINConv(
            nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(),
                       nn.Linear(dim, dim), nn.ReLU()))

        self.conv4 = torch_geometric.nn.GINConv(
            nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(),
                       nn.Linear(dim, dim), nn.ReLU()))

        self.conv5 = torch_geometric.nn.GINConv(
            nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(),
                       nn.Linear(dim, dim), nn.ReLU()))

        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, out_channels)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = self.conv3(x, edge_index)
        x = self.conv4(x, edge_index)
        x = self.conv5(x, edge_index)
        x = torch_geometric.nn.global_add_pool(x, batch)
        x = self.lin1(x).relu()
        x = torch.nn.functional.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return torch.nn.functional.log_softmax(x, dim=-1)


def benchmark(model_nn):
    dataset = TUDataset(root=f'/tmp/MUTAG', name='MUTAG').shuffle()

    train_dataset = dataset[len(dataset) // 10:]
    test_dataset = dataset[:len(dataset) // 10]

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128)

    model = model_nn(dataset.num_features, 32, dataset.num_classes)

    train_stats = train_eval_loop(model, train_loader, test_loader)
    print(train_stats)

def benchmark_catgnn_gin_1():
    benchmark(CatGNN_GIN_1)

def benchmark_catgnn_gin_2():
    benchmark(CatGNN_GIN_2)

def benchmark_catgnn_gin_3():
    benchmark(CatGNN_GIN_3)

def benchmark_pyg_gcn():
    benchmark(PyG_GIN)


if __name__ == '__main__':
    #benchmark_catgnn_gin_1()
    #benchmark_catgnn_gin_2()
    #benchmark_catgnn_gin_3()
    benchmark_pyg_gcn()