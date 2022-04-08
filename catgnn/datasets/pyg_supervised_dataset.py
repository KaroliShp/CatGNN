"""
Adapted from
https://github.com/pyg-team/pytorch_geometric/blob/master/benchmark/kernel/datasets.py
"""

import os.path as osp

import torch

import torch_geometric


class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = torch_geometric.utils.degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data


def get_TU_dataset(name, cleaned=False):
    assert name in ['MUTAG', 'PROTEINS', 'REDDIT-BINARY']

    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
    dataset = torch_geometric.datasets.TUDataset(path, name, cleaned=cleaned)
    dataset.data.edge_attr = None

    if dataset.data.x is None:
        max_degree = 0
        degs = []
        for data in dataset:
            degs += [torch_geometric.utils.degree(data.edge_index[0], dtype=torch.long)]
            max_degree = max(max_degree, degs[-1].max().item())

        if max_degree < 1000:
            dataset.transform = torch_geometric.transforms.OneHotDegree(max_degree)
        else:
            deg = torch.cat(degs, dim=0).to(torch.float)
            mean, std = deg.mean().item(), deg.std().item()
            dataset.transform = NormalizedDegree(mean, std)

    return dataset