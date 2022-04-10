import pytest
from catgnn.integral_transform.mpnn_2 import BaseMPNNLayer_2
import torch


"""
MPNN_2
"""


@pytest.mark.parametrize('E,expected_E', [
    (
        torch.tensor([[0, 1, 1, 2, 2, 3],
                      [1, 0, 2, 1, 3, 2]], dtype=torch.int64),
        torch.tensor([1, 0, 2, 1, 3, 2], dtype=torch.int64),
    ),
])
def test_t_mpnn_2(E, expected_E):
    """
    Test functionality of getting receivers for given E
    Note: edge indices are sender to receiver
    """
    base_layer = BaseMPNNLayer_2()
    output_E = base_layer.t(E)

    assert torch.equal(output_E, expected_E)


@pytest.mark.parametrize('E,expected_E', [
    (
        torch.tensor([[0, 1, 1, 2, 2, 3],
                      [1, 0, 2, 1, 3, 2]], dtype=torch.int64),
        torch.tensor([0, 1, 1, 2, 2, 3], dtype=torch.int64),
    ),
])
def test_s_mpnn_2(E, expected_E):
    """
    Test functionality of getting senders for given E
    Note: edge indices are sender to receiver
    """
    base_layer = BaseMPNNLayer_2()
    output_E = base_layer.s(E)

    assert torch.equal(output_E, expected_E)