import numpy as np


def accuracy(net, X, y):
    if X.ndim == 1:
        X = X.reshape((-1, 1))

    if y.ndim == 1:
        y = y.reshape((-1, 1))

    yhat = net.predict(X)
    return (yhat == y).mean()
