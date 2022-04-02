import pytest
from catgnn.models.generic_mpnn_1 import GenericMPNNLayer
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
def test_generic_mpnn_1(V, E, X):
    expected_output = torch.tensor([[0., 1.], 
                                    [1., 0.], 
                                    [1., 2.], 
                                    [1., 0.]])

    generic_layer = GenericMPNNLayer()
    output = generic_layer(V, E, X)

    assert torch.equal(output, expected_output)