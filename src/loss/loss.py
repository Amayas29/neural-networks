import numpy as np
from .base import Loss
from utils.processing import one_hot_y


class MSELoss(Loss):
    def __init__(self):
        super().__init__()

    def forward(self, y, yhat):
        assert y.shape == yhat.shape, ValueError(
            "Les dimensions de y et yhat ne correspondent pas."
        )

        return np.linalg.norm(y - yhat, axis=1) ** 2

    def backward(self, y, yhat):
        assert y.shape == yhat.shape, ValueError(
            "Les dimensions de y et yhat ne correspondent pas."
        )

        return -2 * (y - yhat)


class CrossEntropie(Loss):
    def __init__(self):
        super().__init__()

    def forward(self, y, yhat):
        assert y.shape == yhat.shape, ValueError(
            "Les dimensions y et yhat ne correspondent pas."
        )

        return (-y * yhat).sum(axis=1)

    def backward(self, y, yhat):
        assert y.shape == yhat.shape, ValueError(
            "Les dimensions y et yhat ne correspondent pas."
        )

        return -y


class CELogSoftmax(Loss):
    def __init__(self):
        super().__init__()

    def forward(self, y, yhat):
        assert y.shape == yhat.shape, ValueError(
            "Les dimensions y et yhat ne correspondent pas."
        )

        return (-y * yhat).sum(axis=1) + np.log(np.exp(yhat).sum(axis=1))

    def backward(self, y, yhat):
        assert y.shape == yhat.shape, ValueError(
            "Les dimensions y et yhat ne correspondent pas."
        )

        return -y + np.exp(yhat) / np.exp(yhat).sum(axis=1).reshape(-1, 1)


class BCELoss(Loss):
    def __init__(self):
        super().__init__()

    def forward(self, y, yhat):
        assert y.shape == yhat.shape, ValueError(
            "Les dimensions y et yhat ne correspondent pas."
        )

        return -(
            y * np.maximum(-100, np.log(yhat + 1e-3))
            + (1 - y) * np.maximum(-100, np.log(1 - yhat + 1e-3))
        )

    def backward(self, y, yhat):
        assert y.shape == yhat.shape, ValueError(
            "Les dimensions y et yhat ne correspondent pas."
        )

        return -(y / (yhat + 1e-3)) + ((1 - y) / (1 - yhat + 1e-3))
