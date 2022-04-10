import pytest
from catgnn.integral_transform.mpnn_1 import BaseMPNNLayer_1
from catgnn.integral_transform.mpnn_2 import BaseMPNNLayer_2
import torch


"""
MPNN_1
"""


@pytest.mark.parametrize('V,V_chosen,E,expected_preimages', [
    (
        torch.tensor([0, 1, 2, 3], dtype=torch.int64),
        torch.tensor([0, 1, 2, 3], dtype=torch.int64),
        torch.tensor([[0, 1, 1, 2, 2, 3],
                      [1, 0, 2, 1, 3, 2]], dtype=torch.int64),
        [
            [ 
                torch.tensor([1, 0, 1], dtype=torch.int64) 
            ],
            [ 
                torch.tensor([0, 1, 0], dtype=torch.int64),
                torch.tensor([2, 1, 3], dtype=torch.int64) 
            ],
            [ 
                torch.tensor([1, 2, 2], dtype=torch.int64),
                torch.tensor([3, 2, 5], dtype=torch.int64) 
            ],
            [ 
                torch.tensor([2, 3, 4], dtype=torch.int64) 
            ],
        ]
    ),
    (
        torch.tensor([0, 1, 2, 3], dtype=torch.int64),
        torch.tensor([0, 2, 3], dtype=torch.int64),
        torch.tensor([[0, 1, 1, 2, 2, 3],
                      [1, 0, 2, 1, 3, 2]], dtype=torch.int64),
        [
            [ 
                torch.tensor([1, 0, 1], dtype=torch.int64) 
            ],
            [ 
                torch.tensor([0, 1, 0], dtype=torch.int64),
                torch.tensor([2, 1, 3], dtype=torch.int64) 
            ],
            [ 
                torch.tensor([1, 2, 2], dtype=torch.int64),
                torch.tensor([3, 2, 5], dtype=torch.int64) 
            ],
            [ 
                torch.tensor([2, 3, 4], dtype=torch.int64) 
            ],
        ]
    ),
    (
        torch.tensor([0, 1, 2, 3], dtype=torch.int64),
        torch.tensor([0, 3], dtype=torch.int64),
        torch.tensor([[0, 1, 1, 2, 2, 3],
                      [1, 0, 2, 1, 3, 2]], dtype=torch.int64),
        [
            [ 
                torch.tensor([1, 0, 1], dtype=torch.int64) 
            ],
            [ 
                torch.tensor([0, 1, 0], dtype=torch.int64),
                torch.tensor([2, 1, 3], dtype=torch.int64) 
            ],
            [ 
                torch.tensor([1, 2, 2], dtype=torch.int64),
                torch.tensor([3, 2, 5], dtype=torch.int64) 
            ],
            [ 
                torch.tensor([2, 3, 4], dtype=torch.int64) 
            ],
        ]
    ),
])
def test_inverse_t_mpnn_1(V, V_chosen, E, expected_preimages):
    base_layer = BaseMPNNLayer_1()
    base_layer._add_edge_indices(E)
    base_layer._set_preimages(V)

    for i in range(V_chosen.shape[0]):
        output = base_layer.t_1(V_chosen[i])
        assert len(output) == len(expected_preimages[V_chosen[i]])
        for j in range(len(output)):
            assert torch.equal(output[j], expected_preimages[V_chosen[i]][j])


"""
MPNN_2
"""


@pytest.mark.parametrize('V,E,expected_E,expected_indices', [
    (
        torch.tensor([0, 1, 2, 3], dtype=torch.int64),
        torch.tensor([[0, 1, 1, 2, 2, 3],
                      [1, 0, 2, 1, 3, 2]], dtype=torch.int64),
        torch.tensor([[0, 1, 1, 2, 2, 3],
                      [1, 0, 2, 1, 3, 2]], dtype=torch.int64),
        torch.tensor([1, 0, 2, 1, 3, 2], dtype=torch.int64),
    ),
    (
        torch.tensor([0, 2, 3], dtype=torch.int64),
        torch.tensor([[0, 1, 1, 2, 2, 3],
                      [1, 0, 2, 1, 3, 2]], dtype=torch.int64),
        torch.tensor([[1, 1, 2, 3],
                      [0, 2, 3, 2]], dtype=torch.int64),
        torch.tensor([0, 2, 3, 2], dtype=torch.int64),
    ),
    (
        torch.tensor([0, 3], dtype=torch.int64),
        torch.tensor([[0, 1, 1, 2, 2, 3],
                      [1, 0, 2, 1, 3, 2]], dtype=torch.int64),
        torch.tensor([[1, 2],
                      [0, 3]], dtype=torch.int64),
        torch.tensor([0, 3], dtype=torch.int64),
    ),
])
def test_inverse_t_mpnn_2(V, E, expected_E, expected_indices):
    """
    Test functionality of finding preimages of given V
    """
    base_layer = BaseMPNNLayer_2()
    base_layer.E = E
    output_E, output_indices = base_layer.t_1(V)

    assert torch.equal(output_E, expected_E)
    assert torch.equal(output_indices, expected_indices)