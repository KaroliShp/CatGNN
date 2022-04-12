import pytest
from catgnn.layers.sgc.sgc_mpnn_2 import SGCLayer_MPNN_2
import torch
import copy


@pytest.mark.parametrize("V,E,X,in_dim,out_dim,expected_X_shape", [
    # Undirected graph
    (
        torch.tensor([0, 1, 2, 3], dtype=torch.int64),
        torch.tensor(
            [(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)], dtype=torch.int64
        ).T,
        torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32),
        2, 2,
        (4,2)
    ),
    # Undirected graph, different features
    (
        torch.tensor([0, 1, 2, 3], dtype=torch.int64),
        torch.tensor(
            [(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)], dtype=torch.int64
        ).T,
        torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32),
        2, 3,
        (4,3)
    ),
    # Two directed edges
    (
        torch.tensor([0, 1, 2, 3], dtype=torch.int64),
        torch.tensor(
            [(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 1)], dtype=torch.int64
        ).T,
        torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32),
        2,2,
        (4,2)
    ),
])
def test_sgc_mpnn_2(V, E, X, in_dim, out_dim, expected_X_shape):
    sgc_layer = SGCLayer_MPNN_2(in_dim, out_dim)
    output = sgc_layer(V, E, X)

    assert output.shape == expected_X_shape


@pytest.mark.parametrize("in_dim,out_dim", [
    (2,2)
])
def test_sgc_mpnn_2_reset_params(in_dim, out_dim):
    sgc_layer = SGCLayer_MPNN_2(in_dim, out_dim)
    before_params = copy.deepcopy([p for p in sgc_layer.parameters()])
    sgc_layer.reset_parameters()
    after_params = copy.deepcopy([p for p in sgc_layer.parameters()])

    assert len(before_params) == len(after_params)
    for i in range(len(before_params)):
        assert not torch.equal(before_params[i], after_params[i])
