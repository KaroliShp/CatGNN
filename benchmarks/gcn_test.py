import argparse
import os.path as osp

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import ChebConv, GCNConv  # noqa
from catgnn.layers.gcn_conv.gcn_mpnn_2 import GCNLayer_MPNN_2
from catgnn.layers.gcn_conv.pyg_custom_gcn import GCNConvCustom

from timeit import default_timer as timer
from datetime import timedelta

parser = argparse.ArgumentParser()
parser.add_argument('--use_gdc', action='store_true',
                    help='Use GDC preprocessing.')
args = parser.parse_args()

dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
data = dataset[0]

if args.use_gdc:
    gdc = T.GDC(self_loop_weight=1, normalization_in='sym',
                normalization_out='col',
                diffusion_kwargs=dict(method='ppr', alpha=0.05),
                sparsification_kwargs=dict(method='topk', k=128,
                                           dim=0), exact=True)
    data = gdc(data)


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        #self.conv1 = GCNConv(dataset.num_features, 16, cached=False)
        #self.conv2 = GCNConv(16, dataset.num_classes, cached=False)

        self.conv1 = GCNLayer_MPNN_2(dataset.num_features, 16)
        self.conv2 = GCNLayer_MPNN_2(16, dataset.num_classes)

        #self.conv1 = GCNConvCustom(dataset.num_features, 16)
        #self.conv2 = GCNConvCustom(16, dataset.num_classes)

        # self.conv1 = ChebConv(data.num_features, 16, K=2)
        # self.conv2 = ChebConv(16, data.num_features, K=2)

    def forward(self):
        """
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
        """

        x, edge_index, _ = data.x, data.edge_index, data.edge_attr
        v = torch.arange(0,torch.max(edge_index)+1)
        x = F.relu(self.conv1(v, edge_index, x))
        x = F.dropout(x, training=self.training)
        x = self.conv2(v, edge_index, x)
        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = Net().to(device), data.to(device)
optimizer = torch.optim.Adam([
    dict(params=model.conv1.parameters(), weight_decay=5e-4),
    dict(params=model.conv2.parameters(), weight_decay=0)
], lr=0.01)  # Only perform weight-decay on first convolution.


def train():
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model()[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()


@torch.no_grad()
def test():
    model.eval()
    logits, accs = model(), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


best_val_acc = test_acc = 0
for epoch in range(1, 101):
    start = timer()
    train()
    end = timer()
    print(f'Time: {timedelta(seconds=end-start)}')

    train_acc, val_acc, tmp_test_acc = test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    if epoch % 10 == 0:
        print(f'Epoch: {epoch:03d}, Train: {train_acc:.4f}, '
            f'Val: {best_val_acc:.4f}, Test: {test_acc:.4f}')