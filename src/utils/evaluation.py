import numpy as np


def accuracy(net, X, y, multi_class=False):
    if X.ndim == 1:
        X = X.reshape((-1, 1))

    if y.ndim == 1:
        y = y.reshape((-1, 1))

    ypred = net(X)

    if multi_class:
        yhat = np.argmax(ypred, axis=1, keepdims=True)
    else:
        neg_class = np.min(y)

        thershold = 0.5
        if neg_class == -1:
            thershold = 0

        yhat = np.where(ypred < thershold, neg_class, 1)

    return (yhat == y).mean()
