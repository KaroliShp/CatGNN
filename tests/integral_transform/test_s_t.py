import pytest
from catgnn.integral_transform.mpnn_1 import BaseMPNNLayer_1
from catgnn.integral_transform.mpnn_2 import BaseMPNNLayer_2
import torch


"""
t
"""


TEST_CASES_T = [
    (
        torch.tensor([[0, 1, 1, 2, 2, 3],
                      [1, 0, 2, 1, 3, 2]], dtype=torch.int64),
        torch.tensor([1, 0, 2, 1, 3, 2], dtype=torch.int64),
    ),
]


@pytest.mark.parametrize('E,expected_E', TEST_CASES_T)
def test_t_mpnn_1(E, expected_E):
    base_layer = BaseMPNNLayer_1()
    base_layer._add_edge_indices(E)
    output_E = []
    for e in range(base_layer.E_indexed.shape[1]):
        output_E.append(base_layer.t(base_layer.E_indexed[:,e]))
    output_E = torch.tensor(output_E, dtype=torch.int64)

    assert torch.equal(output_E, expected_E)


@pytest.mark.parametrize('E,expected_E', TEST_CASES_T)
def test_t_mpnn_2(E, expected_E):
    """
    Test functionality of getting receivers for given E
    Note: edge indices are sender to receiver
    """
    base_layer = BaseMPNNLayer_2()
    output_E = base_layer.t(E)

    assert torch.equal(output_E, expected_E)


"""
s
"""


TEST_CASES_S = [
    (
        torch.tensor([[0, 1, 1, 2, 2, 3],
                      [1, 0, 2, 1, 3, 2]], dtype=torch.int64),
        torch.tensor([0, 1, 1, 2, 2, 3], dtype=torch.int64),
    ),
]


@pytest.mark.parametrize('E,expected_E', TEST_CASES_S)
def test_s_mpnn_1(E, expected_E):
    base_layer = BaseMPNNLayer_1()
    base_layer._add_edge_indices(E)
    output_E = []
    for e in range(base_layer.E_indexed.shape[1]):
        output_E.append(base_layer.s(base_layer.E_indexed[:,e]))
    output_E = torch.tensor(output_E, dtype=torch.int64)

    assert torch.equal(output_E, expected_E)


@pytest.mark.parametrize('E,expected_E', TEST_CASES_S)
def test_s_mpnn_2(E, expected_E):
    """
    Test functionality of getting senders for given E
    Note: edge indices are sender to receiver
    """
    base_layer = BaseMPNNLayer_2()
    output_E = base_layer.s(E)

    assert torch.equal(output_E, expected_E)