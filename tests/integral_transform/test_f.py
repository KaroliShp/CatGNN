import pytest
from catgnn.integral_transform.mpnn_3 import BaseMPNNLayer_3
import torch


"""
MPNN_3
"""


@pytest.mark.parametrize('V,X,expected_X', [
    (
        torch.tensor([0, 1, 2, 3], dtype=torch.int64),
        torch.tensor([[0,0], [0,1], [1,0], [1,1]]),
        torch.tensor([[0,0], [0,1], [1,0], [1,1]])
    ),
    (
        torch.tensor([0, 2, 3], dtype=torch.int64),
        torch.tensor([[0,0], [0,1], [1,0], [1,1]]),
        torch.tensor([[0,0], [1,0], [1,1]])
    ),
    (
        torch.tensor([0, 2, 1, 1, 3], dtype=torch.int64),
        torch.tensor([[0,0], [0,1], [1,0], [1,1]]),
        torch.tensor([[0,0], [1,0], [0,1], [0,1], [1,1]])
    ),
])
def test_f_mpnn_3(V, X, expected_X):
    """
    Test functionality of getting features for vertices
    """
    base_layer = BaseMPNNLayer_3()
    base_layer.X = X
    output_X = base_layer.f(V)

    assert torch.equal(output_X, expected_X)
