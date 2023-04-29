from .module import Module
import numpy as np


class Linear(Module):

    def __init__(self, input_dim, output_dim, bias=False):
        super().__init__()

        self._input_dim = input_dim
        self._output_dim = output_dim

        self._parameters = 2 * (np.random.rand(input_dim, output_dim) - 0.5)

        self._bias = None
        if bias:
            self._bias = np.ones(output_dim)

        self.zero_grad()

    def zero_grad(self):
        self._gradient = np.zeros((self._input_dim, self._output_dim))
        self._bias_gradient = np.zeros(self._output_dim)

    def update_parameters(self, gradient_step=1e-3):
        self._parameters -= gradient_step * self._gradient

        if self._bias is not None:
            self._bias -= gradient_step * self._bias_gradient

    def forward(self, X):
        assert X.shape[1] == self._input_dim, "Erreur - Linear:forward : dimension des entrées"

        assert self._parameters.shape == (self._input_dim,
                                          self._output_dim), "Erreur - Linear:forward : dimension de W"

        out = np.dot(X, self._parameters)

        if self._bias is not None:
            out += self._bias

        return out

    def backward_update_gradient(self, X, delta):
        assert X.shape[1] == self._input_dim, "Erreur - Linear:backward_update_gradient : dimension des entrées"
        assert delta.shape[1] == self._output_dim, "Erreur - Linear:backward_update_gradient : dimension de deltas"
        assert delta.shape[0] == X.shape[0], "Erreur - Linear:backward_update_gradient : dimension 'batch' pour delta et les entrées"

        self._gradient += np.dot(X.T, delta)

        if self._bias is not None:
            self._bias_gradient += np.sum(delta, axis=0)

    def backward_delta(self, X, delta):
        assert delta.shape[1] == self._output_dim, "Erreur - Linear:backward_delta : dimension des entrées delta"
        assert X.shape[1] == self._input_dim, "Erreur - Linear:backward_delta : dimension des entrées input"
        assert delta.shape[0] == X.shape[0], "Erreur - Linear:backward_update_gradient : dimension 'batch' pour delta et les entrées"

        out = np.dot(delta, self._parameters.T)

        if self._bias is not None:
            out += self._bias

        return out
