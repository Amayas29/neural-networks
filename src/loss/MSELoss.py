import numpy as np
from .Loss import Loss


class MSELoss(Loss):

    def forward(self, y, yhat):
        assert y.shape == yhat.shape, "Erreur - MSELoss:forward : tailles de y et yhat"
        return np.linalg.norm(y - yhat) ** 2

    def backward(self, y, yhat):
        assert y.shape == yhat.shape, "Erreur - MSELoss:backward : tailles de y et yhat"
        return -2 * (y - yhat)
