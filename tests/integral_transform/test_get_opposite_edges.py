import pytest
from catgnn.integral_transform.mpnn_2 import BaseMPNNLayer_2
import torch

"""
MPNN_2
"""

@pytest.mark.parametrize('E,expected_E', [
    # Undirected graph
    (
        torch.tensor([[0, 1, 1, 2, 2, 3],
                      [1, 0, 2, 1, 3, 2]], dtype=torch.int64),
        torch.tensor([[1, 0, 2, 1, 3, 2],
                      [0, 1, 1, 2, 2, 3]], dtype=torch.int64)
    ),
    # 2 -> 3, but no 3 -> 2
    (
        torch.tensor([[0, 1, 1, 2, 2],
                      [1, 0, 2, 1, 3]], dtype=torch.int64),
        torch.tensor([[1, 0, 2, 1, -1],
                      [0, 1, 1, 2, -1]], dtype=torch.int64)
    ),
    # 2 -> 3 and 3 -> 1, but no 3 -> 2 and 1 -> 3
    (
        torch.tensor([[0, 1, 1, 2, 2, 3],
                      [1, 0, 2, 1, 3, 1]], dtype=torch.int64),
        torch.tensor([[1, 0, 2, 1, -1, -1],
                      [0, 1, 1, 2, -1, -1]], dtype=torch.int64)
    ),
    # Only directed edges (so no pullback from receivers)
    (
        torch.tensor([[0, 1, 2],
                      [1, 2, 3]], dtype=torch.int64),
        torch.tensor([[-1, -1, -1],
                      [-1, -1, -1]], dtype=torch.int64)
    ),
])
def test_inverse_t_mpnn_2_masked(E, expected_E):
    """
    Test functionality of finding opposite edges for E (with masking)
    """
    base_layer = BaseMPNNLayer_2()
    output_E = base_layer.get_opposite_edges(E, masking_required=True)

    assert torch.equal(output_E, expected_E)


@pytest.mark.parametrize('E,expected_E', [
    # Undirected graph
    (
        torch.tensor([[0, 1, 1, 2, 2, 3],
                      [1, 0, 2, 1, 3, 2]], dtype=torch.int64),
        torch.tensor([[1, 0, 2, 1, 3, 2],
                      [0, 1, 1, 2, 2, 3]], dtype=torch.int64)
    ),
    # 2 -> 3, but no 3 -> 2
    (
        torch.tensor([[0, 1, 1, 2, 2],
                      [1, 0, 2, 1, 3]], dtype=torch.int64),
        torch.tensor([[1, 0, 2, 1, 3],
                      [0, 1, 1, 2, 2]], dtype=torch.int64)
    ),
    # 2 -> 3 and 3 -> 1, but no 3 -> 2 and 1 -> 3
    (
        torch.tensor([[0, 1, 1, 2, 2, 3],
                      [1, 0, 2, 1, 3, 1]], dtype=torch.int64),
        torch.tensor([[1, 0, 2, 1, 3, 1],
                      [0, 1, 1, 2, 2, 3]], dtype=torch.int64)
    ),
    # Only directed edges (so no pullback from receivers)
    (
        torch.tensor([[0, 1, 2],
                      [1, 2, 3]], dtype=torch.int64),
        torch.tensor([[1, 2, 3],
                      [0, 1, 2]], dtype=torch.int64)
    ),
])
def test_inverse_t_mpnn_2_unmasked(E, expected_E):
    """
    Test functionality of finding opposite edges for E (unmasked)
    """
    base_layer = BaseMPNNLayer_2()
    output_E = base_layer.get_opposite_edges(E, masking_required=False)

    assert torch.equal(output_E, expected_E)