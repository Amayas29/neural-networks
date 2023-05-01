from .base import Module
import numpy as np


class Sigmoide(Module):
    def __init__(self):
        super().__init__()

    def zero_grad(self):
        pass  # No gradient to zero

    def update_parameters(self, gradient_step=1e-3):
        pass  # No parametrs to update

    def backward_update_gradient(self, X, delta):
        pass  # No gradient to update

    def forward(self, X):
        return 1 / (1 + np.exp(-X))

    def backward_delta(self, X, delta):
        sig = self(X)
        return sig * (1 - sig) * delta


class TanH(Module):
    def __init__(self):
        super().__init__()

    def zero_grad(self):
        pass  # No gradient to zero

    def update_parameters(self, gradient_step=1e-3):
        pass  # No parametrs to update

    def backward_update_gradient(self, X, delta):
        pass  # No gradient to update

    def forward(self, X):
        return np.tanh(X)

    def backward_delta(self, X, delta):
        return (1 - np.tanh(X) ** 2) * delta


class Softmax(Module):
    def __init__(self):
        super().__init__()

    def zero_grad(self):
        pass  # No gradient to zero

    def update_parameters(self, gradient_step=1e-3):
        pass  # No parametrs to update

    def backward_update_gradient(self, X, delta):
        pass  # No gradient to update

    def forward(self, X):
        exp_X = np.exp(X)
        return exp_X / np.sum(exp_X, axis=1, keepdims=True)

    def backward_delta(self, X, delta):
        softmax = self(X)
        return (softmax * (1 - softmax)) * delta
