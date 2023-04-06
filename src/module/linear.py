from .module import Module
import numpy as np


class Linear(Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()

        self._input_dim = input_dim
        self._output_dim = output_dim
        self._parameters = 2 * \
            (np.random.rand(self._input_dim, self._output_dim) - 0.5)
        self.zero_grad()

    def zero_grad(self):
        self._gradient = np.zeros((self._input_dim, self._output_dim))

    def forward(self, X):
        assert X.shape[1] == self._input_dim, "Erreur - Linear:forward : dimension des entrées"

        assert self._parameters.shape == (self._input_dim,
                                          self._output_dim), "Erreur - Linear:forward : dimension de W"

        return np.dot(X, self._parameters)

    def backward_update_gradient(self, X, delta):
        assert X.shape[1] == self._input_dim, "Erreur - Linear:backward_update_gradient : dimension des entrées"
        assert delta.shape[1] == self._output_dim, "Erreur - Linear:backward_update_gradient : dimension de deltas"
        assert delta.shape[0] == X.shape[0], "Erreur - Linear:backward_update_gradient : dimension 'batch' pour delta et les entrées"

        self._gradient += X.T @ delta

    def backward_delta(self, X, delta):
        assert delta.shape[1] == self._output_dim, "Erreur - Linear:backward_delta : dimension des entrées delta"
        assert input_.shape[1] == self._input_dim, "Erreur - Linear:backward_delta : dimension des entrées input"
        assert delta.shape[0] == X.shape[0], "Erreur - Linear:backward_update_gradient : dimension 'batch' pour delta et les entrées"

        return delta @ self._parameters.T
