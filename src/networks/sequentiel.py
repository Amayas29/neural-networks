import numpy as np


class Sequentiel:
    def __init__(self, modules):
        self.modules = modules
        self.inputs_modules = []

    def add_module(self, module):
        self.modules.append(module)

    def forward(self, X):
        self.inputs_modules = []

        for module in self.modules:
            self.inputs_modules.append(X)
            X = module.forward(X)

        return X

    def backward(self, delta, eps=1e-5):

        for i in range(len(self.modules) - 1, -1, -1):
            self.modules[i].zero_grad()

            X = self.inputs_modules[i]
            self.modules[i].backward_update_gradient(X, delta)
            self.modules[i].update_parameters(eps)

            delta = self.modules[i].backward_delta(X, delta)
