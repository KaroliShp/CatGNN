import os.path as osp

import torch
import torch.nn.functional as F

from torch_geometric.datasets import Planetoid
from torch_geometric.nn import SGConv

from catgnn.layers.sgc.sgc_mpnn_2 import SGCLayer_MPNN_2

from timeit import default_timer as timer
from datetime import timedelta

dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Planetoid(path, dataset)
data = dataset[0]


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = SGConv(dataset.num_features, dataset.num_classes, K=2, cached=False)
        #self.conv1 = SGCLayer_MPNN_2(dataset.num_features, dataset.num_classes, K=2)

    def forward(self):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        return F.log_softmax(x, dim=1)
        """
        x, edge_index = data.x, data.edge_index
        v = torch.arange(0,torch.max(edge_index)+1)
        x = self.conv1(v, edge_index, x)
        return F.log_softmax(x, dim=1)
        """


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = Net().to(device), data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.2, weight_decay=0.005)


def train():
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model()[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()


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