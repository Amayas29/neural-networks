from .base import Module
import numpy as np


class Linear(Module):
    def __init__(self, input_dim, output_dim, bias=False):
        super().__init__()

        self._input_dim = input_dim
        self._output_dim = output_dim
        self.bias = bias

        self._parameters["W"] = 2 * (np.random.rand(input_dim, output_dim) - 0.5)

        if self.bias:
            self._parameters["b"] = np.random.randn(1, self._output_dim)

        self.zero_grad()

    def zero_grad(self):
        self._gradient["W"] = np.zeros_like(self._parameters["W"])

        if self.bias:
            self._gradient["b"] = np.zeros_like(self._parameters["b"])

    def update_parameters(self, gradient_step=1e-3):
        self._parameters["W"] -= gradient_step * self._gradient["W"]

        if self.bias:
            self._parameters["b"] -= gradient_step * self._gradient["b"]

    def forward(self, X):
        assert X.shape[1] == self._input_dim, ValueError(
            "Les dimensions de X doivent être (batch_size, input_dim)"
        )

        out = np.dot(X, self._parameters["W"])

        if self.bias:
            out += self._parameters["b"]

        return out

    def backward_update_gradient(self, X, delta):
        assert X.shape[1] == self._input_dim, ValueError(
            "Les dimensions de X doivent être (batch_size, input_dim)"
        )

        assert delta.shape == (X.shape[0], self._output_dim), ValueError(
            "Delta doit être de dimension (batch_size, output_dim)"
        )

        self._gradient["W"] += np.dot(X.T, delta)

        if self.bias:
            self._gradient["b"] += np.sum(delta, axis=0)

    def backward_delta(self, X, delta):
        assert delta.shape == (X.shape[0], self._output_dim), ValueError(
            "Delta doit être de dimension (batch_size, output_dim)"
        )

        return np.dot(delta, self._parameters["W"].T)
