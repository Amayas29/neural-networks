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
    def __init__(self, nb_classes):
        super().__init__()

        self.nb_classes = nb_classes

    def forward(self, y, yhat):
        # TODO
        assert y.shape == (yhat.shape[0], 1), ValueError(
            "Les dimensions y ne correspondent pas."
        )

        y_one_hot = one_hot_y(y, self.nb_classes)

        indices = np.argmax(y_one_hot, axis=1)

        y_class = yhat[np.arange(len(y)), indices].reshape(-1, 1)

        return -y_class + np.log(np.exp(yhat).sum(axis=1, keepdims=True))

    def backward(self, y, yhat):
        assert y.shape == (yhat.shape[0], 1), ValueError(
            "Les dimensions y ne correspondent pas."
        )

        y_one_hot = one_hot_y(y, self.nb_classes)

        return yhat - y_one_hot
