import os.path as osp

import torch

import torch_geometric.transforms as T
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree


class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data


def get_dataset(name, sparse=True, cleaned=False):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
    dataset = TUDataset(path, name, cleaned=cleaned)
    dataset.data.edge_attr = None

    if dataset.data.x is None:
        print('NO DATA.X')
        max_degree = 0
        degs = []
        for data in dataset:
            degs += [degree(data.edge_index[0], dtype=torch.long)]
            max_degree = max(max_degree, degs[-1].max().item())

        if max_degree < 1000:
            dataset.transform = T.OneHotDegree(max_degree)
        else:
            deg = torch.cat(degs, dim=0).to(torch.float)
            mean, std = deg.mean().item(), deg.std().item()
            dataset.transform = NormalizedDegree(mean, std)

    return dataset