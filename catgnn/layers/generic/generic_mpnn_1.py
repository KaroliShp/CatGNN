import torch

from catgnn.integral_transform.mpnn_1 import BaseMPNNLayer_1


class GenericMPNNLayer_1(BaseMPNNLayer_1):
    """
    Generic MPNN layer using standard (backwards) implementation with BaseMPNNLayer_1.
    Kernel simply leaves the pulled node features unchanged and propagates them further.
    Used for basic testing purposes.
    """

    def __init__(self):
        super().__init__()

    def forward(self, V, E, X):
        out = self.transform_backwards(V, E, X)
        return out

    def define_pullback(self, f):
        def pullback(e):
            return f(self.s(e))

        return pullback

    def define_kernel(self, pullback):
        def kernel(e):
            return pullback(e)

        return kernel

    def define_pushforward(self, kernel):
        def pushforward(v):
            pE = self.t_1(v)

            # Now we need to apply edge_messages function for each element of pE
            bag_of_messages = []
            for e in pE:
                bag_of_messages.append(kernel(e))
            return bag_of_messages

        return pushforward

    def define_aggregator(self, pushforward):
        def aggregator(v):
            total = 0
            for val in pushforward(v):
                total += val
            return total

        return aggregator

    def update(self, X, output):
        return output