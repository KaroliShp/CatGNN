import pytest
from catgnn.layers.generic.generic_mpnn_3 import GenericMPNNLayer_3, GenericFactoredMPNNLayer_3, GenericMPNNLayer_3_Forwards
from catgnn.typing import *
import torch


TEST_GRAPHS = [
    # Normal undirected graph
    (
        torch.tensor([0, 1, 2, 3], dtype=torch.int64),
        torch.tensor([(0,1), (1,0),
                      (1,2), (2,1), 
                      (2,3), (3,2) ], dtype=torch.int64).T,
        torch.tensor([[0,0], [0,1], [1,0], [1,1]]),
        torch.tensor([[0, 1], [1, 0], [1, 2], [1, 0]])
    ),
    # Two directed edges
    (
        torch.tensor([0, 1, 2, 3], dtype=torch.int64),
        torch.tensor([(0,1), (1,0),
                      (1,2), (2,1), 
                      (2,3), (3,1) ], dtype=torch.int64).T,
        torch.tensor([[0,0], [0,1], [1,0], [1,1]]),
        torch.tensor([[0, 1], [1, 0], [1, 2], [0, 1]])
    )
]

TEST_GRAPHS_FACTORED = [
    # Normal undirected graph
    (
        torch.tensor([0, 1, 2, 3], dtype=torch.int64),
        torch.tensor([(0,1), (1,0),
                      (1,2), (2,1), 
                      (2,3), (3,2) ], dtype=torch.int64).T,
        torch.tensor([[0,0], [0,1], [1,0], [1,1]]),
        torch.tensor([[0, 1], [1, 2], [3, 2], [2, 1]])
    ),
    # Two directed edges
    (
        torch.tensor([0, 1, 2, 3], dtype=torch.int64),
        torch.tensor([(0,1), (1,0),
                      (1,2), (2,1), 
                      (2,3), (3,1) ], dtype=torch.int64).T,
        torch.tensor([[0,0], [0,1], [1,0], [1,1]]),
        torch.tensor([[0, 1], [1, 2], [3, 2], [1, 2]])
    )
]


@pytest.mark.parametrize('V,E,X,expected_output', TEST_GRAPHS)
def test_generic_mpnn_3(V, E, X, expected_output):
    """
    Technically this is an integration test (since it tests the whole pipeline)
    """
    generic_layer = GenericMPNNLayer_3()
    output = generic_layer(V, E, X)

    assert torch.equal(output, expected_output)


@pytest.mark.parametrize('V,E,X,expected_output', TEST_GRAPHS_FACTORED)
def test_generic_factored_mpnn_3(V, E, X, expected_output):
    """
    Technically this is an integration test (since it tests the whole pipeline)
    """
    generic_layer = GenericFactoredMPNNLayer_3()
    output = generic_layer(V, E, X)

    assert torch.equal(output, expected_output)


@pytest.mark.parametrize('V,E,X,expected_output', TEST_GRAPHS)
def test_generic_mpnn_3_forwards(V, E, X, expected_output):
    """
    Technically this is an integration test (since it tests the whole pipeline)
    """
    generic_layer = GenericMPNNLayer_3_Forwards()
    output = generic_layer(V, E, X)

    assert torch.equal(output, expected_output)