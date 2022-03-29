import torch


def get_degrees(V, E):
    return torch.zeros(V.shape[0], dtype=E.dtype).scatter_add_(0, E.T[0], torch.ones(E.T[0].shape, dtype=E.dtype))


def add_self_loops(V, E):
    return torch.cat((E,torch.arange(V.shape[0]).repeat(2,1)), dim=1)