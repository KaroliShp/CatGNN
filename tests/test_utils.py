import pytest
from catgnn.utils import get_degrees, add_self_loops
import torch


@pytest.mark.parametrize(
    "V,E,expected_output",
    [
        (
            torch.tensor([0, 1, 2, 3], dtype=torch.int64),
            torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.int64),
            torch.tensor([1, 2, 2, 1], dtype=torch.int64),
        ),
        (
            torch.tensor([0, 1, 2, 3], dtype=torch.int64),
            torch.tensor([[0, 1, 2, 3], [1, 2, 1, 2]], dtype=torch.int64),
            torch.tensor([0, 2, 2, 0], dtype=torch.int64),
        ),
    ],
)
def test_get_degrees(V, E, expected_output):
    output = get_degrees(V, E)

    assert torch.equal(output, expected_output)


@pytest.mark.parametrize(
    "V,E,expected_output",
    [
        (
            torch.tensor([0, 1, 2, 3], dtype=torch.int64),
            torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.int64),
            torch.tensor(
                [[0, 1, 1, 2, 2, 3, 0, 1, 2, 3], [1, 0, 2, 1, 3, 2, 0, 1, 2, 3]],
                dtype=torch.int64,
            ),
        ),
    ],
)
def test_add_self_loops(V, E, expected_output):
    output = add_self_loops(V, E)

    assert torch.equal(output, expected_output)
