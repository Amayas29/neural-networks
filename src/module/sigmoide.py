from .module import Module
import numpy as np


class Sigmoide(Module):

    def __init__(self):
        super().__init__()

    def zero_grad(self):
        return

    def update_parameters(self):
        return

    def forward(self, X):
        return self.__sigmoid(X)

    def backward_update_gradient(self, X, delta):
        return

    def backward_delta(self, X, delta):
        sig = self.__sigmoid(x)
        return sig * (1 - sig)

    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
