import pytest
from catgnn.integral_transform.mpnn_3 import BaseMPNNLayer_3
import torch


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
def test_inverse_t_mpnn_3(E, expected_E):
    """
    Test functionality of finding opposite edges for E
    """
    base_layer = BaseMPNNLayer_3()
    output_E = base_layer.get_opposite_edges(E)

    assert torch.equal(output_E, expected_E)