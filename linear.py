from abstract import Loss, Module
import numpy as np


class MSELoss(Loss):
    def forward(self, y, yhat):
        assert y.shape == yhat.shape, "Erreur - MSELoss:forward : tailles de y et yhat"
        return np.linalg.norm(y - yhat) ** 2

    def backward(self, y, yhat):
        assert y.shape == yhat.shape, "Erreur - MSELoss:backward : tailles de y et yhat"
        return -2 * (y - yhat)


class Linear(Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()

        self._input_dim = input_dim
        self._output_dim = output_dim
        super()._parameters = np.zeros((input_dim, output_dim))
        self.zero_grad()

    def zero_grad(self):
        super()._gradient = np.zeros((self._input_dim, self._output_dim))

    def forward(self, X):
        assert X.shape[1] == self._input_dim, "Erreur - Linear:forward : dimension des entrées"

        assert super()._parameters.shape == (self._input_dim,
                                             self._output_dim), "Erreur - Linear:forward : dimension de W"

        return np.dot(X, super()._parameters)

    def backward_update_gradient(self, input_, delta):
        super()._gradient += np.dot(input_.T, delta)

    def backward_delta(self, input_, delta):
        assert delta.shape[1] == self._output_dim, "Erreur - Linear:backward_delta : dimension des entrées delta"
        assert input_.shape[1] == self._input_dim, "Erreur - Linear:backward_delta : dimension des entrées input"
        return np.dot(delta, self._parameters.T)
