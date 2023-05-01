import numpy as np


def one_hot_y(y, nb_classes):
    y = y.reshape(-1)
    min_y = np.min(y)
    N = y.shape[0]

    y_shift = y - min_y

    y_oh = np.zeros((N, nb_classes), dtype="int")
    y_oh[np.arange(N), y_shift] = 1

    return y_oh
