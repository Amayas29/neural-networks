import numpy as np


def generate_linear_data(n_samples=100, a=1.0, b=0.0, sigma=0.2):
    X = np.random.rand(n_samples, 1) * 2.0
    y = a * X + b + np.random.randn(n_samples, 1) * sigma

    X = np.concatenate([X, y], axis=1)
    return X


def generate_data_gauss(n_samples, means, sigmas, labels=None):
    n_classes = len(means)
    if labels is None:
        labels = np.arange(n_classes)

    assert len(labels) == n_classes, ValueError(
        "Le nombre d'étiquettes ne correspond pas au nombre de classes."
    )

    X_list = []
    y_list = []

    for mean, sigma, label in zip(means, sigmas, labels):
        X = np.random.multivariate_normal(mean, sigma, n_samples)
        y = np.full(n_samples, label)

        X_list.append(X)
        y_list.append(y)

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    index = np.arange(n_samples * n_classes)
    np.random.shuffle(index)

    return X[index], y[index]
