from collections import defaultdict
import numpy as np


class Module(object):
    def __init__(self, bias=False):
        self._parameters = defaultdict(None)
        self._gradient = defaultdict(None)
        self.bias = bias

    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)

    def __str__(self):
        return f"{self.__class__.__name__}()"

    def __init_parameters__(self, init, W_shape, b_shape=(1, 1)):
        if init == "uniform":
            self._parameters["W"] = np.random.uniform(-1, 1, W_shape) * 0.4
            self._parameters["b"] = np.random.uniform(0, 1, b_shape)

        elif init == "normal":
            self._parameters["W"] = np.random.normal(0, 0.1, W_shape)
            self._parameters["b"] = np.random.normal(0, 0.1, b_shape)

        elif init == "xavier":
            stddev = np.sqrt(2 / (input_dim + output_dim))
            self._parameters["W"] = np.random.normal(0, stddev, W_shape)
            self._parameters["b"] = np.random.normal(0, stddev, b_shape)

        elif init == "he":
            stddev = np.sqrt(2 / input_dim)
            self._parameters["W"] = np.random.normal(0, stddev, W_shape)
            self._parameters["b"] = np.random.normal(0, stddev, b_shape)

        else:
            self._parameters["W"] = np.random.randn(*W_shape)
            self._parameters["b"] = np.random.randn(*b_shape)

        if not self.bias:
            self._parameters["b"] = None

    def update_parameters(self, gradient_step=1e-3):
        """
        Mets à jour les paramètres en utilisant le gradient qui a été calculé et le pas de gradient_step.
        """
        raise NotImplementedError()

    def zero_grad(self):
        """
        Annule le gradient
        """
        raise NotImplementedError()

    def forward(self, X):
        """
        Calcule la passe forward
        """
        raise NotImplementedError()

    def backward_update_gradient(self, X, delta):
        """
        Mets à jour la valeur du gradient
        """
        raise NotImplementedError()

    def backward_delta(self, X, delta):
        """
        Calcule la dérivée de l'erreur par rapport aux entrées.
        """
        raise NotImplementedError()
