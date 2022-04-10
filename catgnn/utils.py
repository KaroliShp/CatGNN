import torch


# Note that degrees are indegrees (incoming into recievers)
def get_degrees(V, E):
    return torch.zeros(V.shape[0], dtype=torch.int64).scatter_add_(
        0, E[1], torch.ones(E.shape[1], dtype=torch.int64)
    )


def add_self_loops(V, E):
    return torch.cat((E, torch.arange(V.shape[0]).repeat(2, 1)), dim=1)
