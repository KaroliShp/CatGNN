# CatGNN 🐱

[![Tests](https://github.com/KaroliShp/CatGNN/actions/workflows/actions.yml/badge.svg?branch=master)](https://github.com/KaroliShp/CatGNN/actions/workflows/actions.yml)

Prototype for a category theory-based GNN library. Implementation of [Graph Neural Networks are Dynamic Programmers (Dudzik and Veličković, 2022)](https://arxiv.org/abs/2203.15544) submitted as coursework for *L45: Representation Learning on Graphs and Networks* course at Cambridge.

The goal of CatGNN is to provide a generic GNN template using a new set of primitives coming from category theory and abstract algebra. Similarly to PyTorch Geomtric, user only needs to provide implementations of the new primitives to implement any GNN.

## Setup

CatGNN was developed on `Python 3.9.12`. To test on GPU, simply import the `gpu_tests.ipynb` notebook to Google Colab and follow the instructions. To test locally, since it is not an official package yet, you can use `environment.yml` with conda. Alternatively:

```bash
$ conda create -n catgnn python=3.9.12
$ conda activate catgnn
$ git clone https://github.com/KaroliShp/CatGNN.git
$ cd CatGNN
$ pip install -r requirements.txt
```

Make sure that `torch` version is at least 1.10, as one of the `BaseMPNNLayer` methods uses `torch.isin()` method which is only available from 1.10.

## Implementation details

For an in-depth explanation of the library, refer to the mini-project report `report.pdf`. 

For benchmarking, see `gpu_tests.ipynb` notebook.

You can test most of the code (as much as custom user layers are testable) using `pytest`.

We generally use [Python Black](https://github.com/psf/black) for code formatting.

For development history, see Issues and Projects tabs.

## Basic example

We can implement a basic Message-Passing GNN (MPNN) layer that applies a simple linear transformation to sender features and uses standard pullback & pushforward operators as follows:

```python
import torch
import torch_scatter

from catgnn.integral_transform.mpnn_2 import BaseMPNNLayer_2


# Custom layer must extend one of the base classes (BaseMPNNLayer_2)
# BaseMPNNLayer_2 is a subclass of torch.nn.Module
class BasicMPNNLayer(BaseMPNNLayer_2):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        # USER IMPLEMENTATION
        self.mlp = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, V, E, X):
        # USER IMPLEMENTATION
        # Perform integral transform by passing V, E and X
        return self.transform_backwards(V, E, X, kernel_factor=False)

    def define_pullback(self, f):
        def pullback(E):
            # s*(e) = f(s(e)) : E -> V
            # USER IMPLEMENTATION
            return f(self.s(E))

        return pullback

    def define_kernel(self, pullback):
        def kernel(E):
            # k(e) : E -> R
            # USER IMPLEMENTATION
            return self.mlp(pullback(E))

        return kernel

    def define_pushforward(self, kernel):
        def pushforward(V):
            # t_*(v) : V -> N[R]
            # USER IMPLEMENTATION
            E, bag_indices = self.t_1(V)
            return kernel(E), bag_indices

        return pushforward

    def define_aggregator(self, pushforward):
        def aggregator(V):
            # \oplus : V -> R
            # USER IMPLEMENTATION
            edge_messages, bag_indices = pushforward(V)
            aggregated = torch_scatter.scatter_add(
                edge_messages.T, bag_indices.repeat(edge_messages.T.shape[0], 1)
            ).T
            return aggregated[V]

        return aggregator

    def update(self, X, output):
        # USER IMPLEMENTATION
        return output
```