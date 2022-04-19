import torch
from torch import Tensor


def get_degrees(V: Tensor, E: Tensor) -> Tensor:
    """
    Get degrees of edges in E. Note that degrees are indegrees (incoming into recievers)

    Args:
        V (Tensor): set of nodes
        E (Tensor): set of edges of shape (2,-1) in sender-to-receiver format

    Returns:
        Tensor: degrees of all nodes
    """    
    return torch.zeros(V.shape[0], dtype=torch.int64, device=V.device).scatter_add_(
        0, E[1], torch.ones(E.shape[1], dtype=torch.int64, device=E.device)
    )


def add_self_loops(V: Tensor, E: Tensor) -> Tensor:
    """
    Add self loops to given set of edges

    Args:
        V (Tensor): set of nodes
        E (Tensor): set of edges of shape (2,-1) in sender-to-receiver format

    Returns:
        Tensor: set of edges of shape (2,-1) in sender-to-receiver format with extra
        self edges for all nodes
    """
    return torch.cat((E, torch.arange(V.shape[0], device=E.device).repeat(2, 1)), dim=1)
