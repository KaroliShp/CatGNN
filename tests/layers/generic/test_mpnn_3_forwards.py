import pytest
from catgnn.layers.generic.generic_mpnn_3 import GenericMPNNLayer_3_Forwards
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
def test_generic_mpnn_1_forwards(V, E, X):
    """
    Technically this is an integration test (since it tests the whole pipeline)
    """
    expected_output = torch.tensor([[0, 1], 
                                    [1, 0], 
                                    [1, 2], 
                                    [1, 0]])

    generic_layer = GenericMPNNLayer_3_Forwards()
    output = generic_layer(V, E, X)

    assert torch.equal(output, expected_output)