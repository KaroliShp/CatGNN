import torch
from torch import nn
from catgnn.typing import *


class BaseMPNNLayer_1(nn.Module):

    def __init__(self):
        super(BaseMPNNLayer_1, self).__init__()

    """
    Directed graph span construction
    """

    def _sender_set(self, E):
        self._s_set = {}
        for i in range(0, E.shape[0]):
            self._s_set[tuple(E[i])] = E[i][1]

    def s(self, e: Type_E) -> Type_V:
        return self._s_set[tuple(e)]

    def _receiver_set(self, E):
        self._t_set = {}
        for i in range(0, E.shape[0]):
            self._t_set[tuple(E[i])] = E[i][0]

    def t(self, e: Type_E) -> Type_V:
        return self._t_set[tuple(e)]

    def _receiver_preimage_set(self, E):
        self._t_1_set = {}
        for i in range(0, E.shape[0]):
            if E[i][0] in self._t_1_set:
                self._t_1_set[E[i][0]].append(E[i])
            else:
                self._t_1_set[E[i][0]] = [E[i]]
    
    def t_1(self, v: Type_V) -> List[Type_E]:
        return self._t_1_set[v]


    """
    Other building blocks for implementing primitive operations
    """


    def _feature_matrix_set(self, V: List[Type_V], X: Type_R):
        self._f_set = {}
        for i in range(0, V.shape[0]):
            assert V[i] not in self._f_set
            self._f_set[V[i]] = X[i]
    
    def f(self, v: Type_V) -> Type_R:
        return self._f_set[v]


    """
    Integral transform primitives
    """

    def define_pullback(self, f: Type_V_R) -> Type_E_R:
        def pullback(e: Type_E) -> Type_R:
            raise NotImplementedError
        return pullback

    def define_kernel(self, er: Type_E_R) -> Type_E_R:
        def kernel_transformation(e: Type_E) -> Type_R:
            raise NotImplementedError
        return kernel_transformation

    def define_pushforward(self, edge_messages: Type_E_R) -> Type_V_NR:
        def pushforward(v: Type_V) -> Type_NR:
            raise NotImplementedError
        return pushforward
    
    def define_aggregator(self, bag_of_values: Type_V_NR) -> Type_V_R:
        def aggregator(v: Type_V) -> Type_R:
            raise NotImplementedError
        return aggregator

    def pipeline(self, V: List[Type_V], E: List[Type_E], X: Type_R):
        # Set the span diagram (sender, receiver functions, node features)
        self._receiver_set(E)
        self._receiver_preimage_set(E)
        self._sender_set(E)
        
        # Set the feature function f : V -> R
        self._feature_matrix_set(V, X)

        # Prepare pipeline
        pullback = self.define_pullback(self.f) # E -> R
        kernel_transformation = self.define_kernel(pullback) # E -> R
        pushforward = self.define_pushforward(kernel_transformation) # V -> N[R]
        aggregator = self.define_aggregator(pushforward) # V -> R

        # Apply the pipeline to each node in the graph
        updated_features = torch.Tensor()
        for v in V:
            updated_features = torch.hstack((updated_features,aggregator(v)))
            #print(updated_features.shape)
        #print(X.shape)
        return updated_features.reshape(X.shape[0],-1)