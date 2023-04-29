from .module import Module
import numpy as np


class TanH(Module):

    def __init__(self):
        super().__init__()

    def zero_grad(self):
        return

    def update_parameters(self, gradient_step=1e-3):
        return

    def forward(self, X):
        return np.tanh(X)

    def backward_update_gradient(self, X, delta):
        return

    def backward_delta(self, X, delta):
        return (1 - np.tanh(X) ** 2) * delta
