import pytest
from catgnn.integral_transform.mpnn_2 import BaseMPNNLayer_2
import torch


"""
MPNN_2
"""


@pytest.mark.parametrize('V,E,expected_E,expected_indices', [
    (
        torch.tensor([0, 1, 2, 3], dtype=torch.int64),
        torch.tensor([[0, 1, 1, 2, 2, 3],
                      [1, 0, 2, 1, 3, 2]], dtype=torch.int64),
        torch.tensor([[0, 1, 1, 2, 2, 3],
                      [1, 0, 2, 1, 3, 2]], dtype=torch.int64),
        torch.tensor([0, 1, 1, 2, 2, 3], dtype=torch.int64),
    ),
    (
        torch.tensor([0, 2, 3], dtype=torch.int64),
        torch.tensor([[0, 1, 1, 2, 2, 3],
                      [1, 0, 2, 1, 3, 2]], dtype=torch.int64),
        torch.tensor([[0, 2, 2, 3],
                      [1, 1, 3, 2]], dtype=torch.int64),
        torch.tensor([0, 2, 2, 3], dtype=torch.int64),
    ),
    (
        torch.tensor([0, 3], dtype=torch.int64),
        torch.tensor([[0, 1, 1, 2, 2, 3],
                      [1, 0, 2, 1, 3, 2]], dtype=torch.int64),
        torch.tensor([[0, 3],
                      [1, 2]], dtype=torch.int64),
        torch.tensor([0, 3], dtype=torch.int64),
    ),
])
def test_inverse_t_mpnn_2(V, E, expected_E, expected_indices):
    """
    Test functionality of finding preimages of given V
    """
    base_layer = BaseMPNNLayer_2()
    base_layer.E = E
    output_E, output_indices = base_layer.t_1(V)

    assert torch.equal(output_E, expected_E)
    assert torch.equal(output_indices, expected_indices)