import numpy as np
from .loss import Loss


class MSELoss(Loss):

    def forward(self, y, yhat):
        """
        @param y    : array(batch * d)
        @param yhat : array(batch * d)

        @return array(batch)
        """

        assert y.shape == yhat.shape, "Erreur - MSELoss:forward : tailles de y et yhat"
        return np.linalg.norm(y - yhat, axis=1) ** 2

    def backward(self, y, yhat):
        """
        @param y    : array(batch * d)
        @param yhat : array(batch * d)

        @return array(batch * d)
        """
        assert y.shape == yhat.shape, "Erreur - MSELoss:backward : tailles de y et yhat"
        return -2 * (y - yhat)
