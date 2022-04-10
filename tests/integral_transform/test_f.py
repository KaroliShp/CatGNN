import pytest
from catgnn.integral_transform.mpnn_1 import BaseMPNNLayer_1
from catgnn.integral_transform.mpnn_2 import BaseMPNNLayer_2
import torch


TEST_CASES = [
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
]


@pytest.mark.parametrize('V,X,expected_X', TEST_CASES)
def test_f_mpnn_1(V, X, expected_X):
    base_layer = BaseMPNNLayer_1()
    base_layer.X = X
    output_X = base_layer.f(V)

    assert torch.equal(output_X, expected_X)


@pytest.mark.parametrize('V,X,expected_X', TEST_CASES)
def test_f_mpnn_2(V, X, expected_X):
    base_layer = BaseMPNNLayer_2()
    base_layer.X = X
    output_X = base_layer.f(V)

    assert torch.equal(output_X, expected_X)
