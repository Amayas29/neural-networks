from .module import Module
import numpy as np


class Sigmoide(Module):

    def __init__(self):
        super().__init__()

    def zero_grad(self):
        return

    def update_parameters(self, gradient_step=1e-3):
        return

    def forward(self, X):
        return self.__sigmoid(X)

    def backward_update_gradient(self, X, delta):
        return

    def backward_delta(self, X, delta):
        return (np.exp(-X) / (1 + np.exp(-X))**2) * delta

    def __sigmoid(self, X):
        return 1 / (1 + np.exp(-X))
