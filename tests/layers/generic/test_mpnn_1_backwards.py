import pytest
from catgnn.layers.generic.generic_mpnn_1 import GenericMPNNLayer_1, GenericFactoredMPNNLayer_1
from catgnn.typing import *
import torch


@pytest.mark.parametrize('V,E,X', [
    (
        torch.tensor([0, 1, 2, 3], dtype=torch.int64),
        torch.tensor([(0,1), (1,0),
                      (1,2), (2,1), 
                      (2,3), (3,2) ], dtype=torch.int64).T,
        torch.tensor([[0,0], [0,1], [1,0], [1,1]])
    )
])
def test_generic_mpnn_1_backwards(V, E, X):
    """
    Technically this is an integration test (since it tests the whole pipeline)
    """
    expected_output = torch.tensor([[0., 1.], 
                                    [1., 0.], 
                                    [1., 2.], 
                                    [1., 0.]])

    generic_layer = GenericMPNNLayer_1()
    output = generic_layer(V, E, X)

    assert torch.equal(output, expected_output)


@pytest.mark.parametrize('V,E,X', [
    (
        torch.tensor([0, 1, 2, 3], dtype=torch.int64),
        torch.tensor([(0,1), (1,0),
                      (1,2), (2,1), 
                      (2,3), (3,2) ], dtype=torch.int64).T,
        torch.tensor([[0,0], [0,1], [1,0], [1,1]])
    )
])
def test_generic_factored_mpnn_1_backwards(V, E, X):
    """
    Technically this is an integration test (since it tests the whole pipeline)
    """
    expected_output = torch.tensor([[0., 1.], 
                                    [1., 0.], 
                                    [1., 2.], 
                                    [1., 0.]])

    generic_factored_layer = GenericFactoredMPNNLayer_1()
    output = generic_factored_layer(V, E, X)

    assert torch.equal(output, expected_output)