import numpy as np


class Sequentiel:
    def __init__(self, modules, classes_type="0/1"):
        self.modules = modules
        self.inputs_modules = []

        self.classes_type = classes_type

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

        if self.classes_type == "multi":
            return np.argmax(ypred, axis=1, keepdims=True)

        neg_class = -1
        th = 0
        if self.classes_type == "0/1":
            neg_class = 0
            th = 0.5

        return np.where(ypred < th, neg_class, 1)

    def score(self, X, y):
        if X.ndim == 1:
            X = X.reshape((-1, 1))

        if y.ndim == 1:
            y = y.reshape((-1, 1))

        if y.shape[1] > 1:
            y = np.argmax(y, axis=1, keepdims=True)

        yhat = self.predict(X)
        return (yhat == y).mean()
