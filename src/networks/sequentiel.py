import numpy as np


class Sequentiel:
    def __init__(self, modules, neg_class=-1):
        self.modules = modules
        self.inputs_modules = []

        self.classes = [neg_class, 1]

    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)

    def forward(self, X):
        self.inputs_modules = []

        for module in self.modules:
            self.inputs_modules.append(X)
            X = module.forward(X)
        return X

    def backward(self, delta):
        for i in range(len(self.modules) - 1, -1, -1):
            X = self.inputs_modules[i]
            self.modules[i].backward_update_gradient(X, delta)
            delta = self.modules[i].backward_delta(X, delta)

    def update_parameters(self, eps=1e-3):
        for module in self.modules:
            module.update_parameters(gradient_step=eps)

    def zero_grad(self):
        for module in self.modules:
            module.zero_grad()

    def predict(self, X):
        ypred = self(X)

        thershold = 0.5
        if self.classes[0] == -1:
            thershold = 0

        return np.where(ypred < thershold, self.classes[0], self.classes[1])
