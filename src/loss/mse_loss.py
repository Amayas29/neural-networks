import numpy as np
from .loss import Loss


class MSELoss(Loss):

    def forward(self, y, yhat):
        assert y.shape == yhat.shape, \
            "Erreur - MSELoss:forward - Les dimensions de y et yhat ne correspondent pas."

        return np.linalg.norm(y - yhat, axis=1) ** 2

    def backward(self, y, yhat):
        assert y.shape == yhat.shape, \
            "Erreur - MSELoss:backward - Les dimensions de y et yhat ne correspondent pas."

        return -2 * (y - yhat)
