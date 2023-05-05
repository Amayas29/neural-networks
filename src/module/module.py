from numpy.lib.stride_tricks import sliding_window_view
from .base import Module
import numpy as np


class Linear(Module):
    def __init__(self, input_dim, output_dim, bias=False, init="uniform"):
        super().__init__(bias)

        self._input_dim = input_dim
        self._output_dim = output_dim
        self.bias = bias

        self.__init_parameters__(
            init, (input_dim, output_dim), (1, output_dim))
        self.zero_grad()

    def __str__(self):
        return f"Linear({self._input_dim}, {self._output_dim})"

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


class Conv1D(Module):
    def __init__(self, k_size, chan_in, chan_out, stride, bias=False, init="uniform"):
        super().__init__(bias)

        self._k_size = k_size
        self._chan_in = chan_in
        self._chan_out = chan_out
        self._stride = stride

        self.__init_parameters__(init, (k_size, chan_in, chan_out), (chan_out))
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
        assert X.shape[2] == self._chan_in, ValueError(
            "Les dimensions de X doivent être (batch, lenght, chan_in)"
        )

        batch_size, length, chan_in = X.shape
        dout = (length - self._k_size) // self._stride + 1
        output = np.zeros((batch_size, dout, self._chan_out))

        for i in range(dout):
            window = X[:, i * self._stride: i * self._stride + self._k_size, :]
            output[:, i, :] = np.tensordot(
                window, self._parameters["W"], axes=([1, 2], [0, 1])
            )

        if self.bias:
            output += self._parameters["b"]

        return output

    def backward_update_gradient(self, X, delta):
        batch_size, length, chan_in = X.shape
        dout = (length - self._k_size) // self._stride + 1

        for i in range(dout):
            window = X[:, i * self._stride: i * self._stride + self._k_size, :]
            self._gradient["W"] += np.tensordot(
                delta[:, i, :], window, axes=([0], [0]))

        if self.bias:
            self._gradient["b"] += np.sum(delta, axis=(0, 1))

    def backward_delta(self, X, delta):
        batch_size, length, chan_in = X.shape
        dout = (length - self._k_size) // self._stride + 1
        delta_prev = np.zeros_like(X)

        for i in range(dout):
            window = X[:, i * self._stride: i * self._stride + self._k_size, :]
            delta_prev[:, i * self._stride: i * self._stride + self._k_size, :] += np.tensordot(
                delta[:, i, :], self._parameters["W"], axes=([1], [0])
            )

        return delta_prev
