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
            y * np.log(np.clip(yhat, 1e-10, 1))
            + (1 - y) * np.log(np.clip(1 - yhat, 1e-10, 1))
        )

    def backward(self, y, yhat):
        assert y.shape == yhat.shape, ValueError(
            "Les dimensions y et yhat ne correspondent pas."
        )

        return -y / np.clip(yhat, 1e-10, 1) + (1 - y) / np.clip(1 - yhat, 1e-10, 1)
