import numpy as np


class Sequentiel:
    def __init__(self, list_modules=[]):
        self.modules = list_modules
        self.inputs_modules = []

    def add_module(self, module):
        self.modules.append(module)

    def forward(self, X):
        self.inputs_modules = []
        for module in self.modules:
            self.inputs_modules.append(X)
            X = module.self.forward(X)
        return X

    def backward(self, delta, eps=1e-5):

        for i in range(len(self.modules) - 1, -1, -1):

            self.module[i].zero_grad()
            x = self.inputs_modules[i]

            self.module[i].backward_update_gradient(x, delta)
            self.module[i].update_parameters(eps)
            delta = self.module[i].backward_delta(x, delta)
