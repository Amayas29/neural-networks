import numpy as np


def one_hot_y(y, nb_classes):
    y = y.reshape(-1)
    min_y = np.min(y)
    N = y.shape[0]

    y_shift = y - min_y

    y_oh = np.zeros((N, nb_classes), dtype="int")
    y_oh[np.arange(N), y_shift] = 1

    return y_oh


def normalisation(data):
    dt = data.copy()

    for i in range(data.shape[1]):
        mini = np.min(data[:, i])
        maxi = np.max(data[:, i])
        if maxi == mini:
            if maxi > 0:
                dt[:, i] = 1
            else:
                dt[:, i] = 0
        else:
            dt[:, i] = (data[:, i] - mini) / (maxi - mini)

    return dt.astype("float64")
