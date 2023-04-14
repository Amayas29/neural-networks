import numpy as np


def generate_linear_data(n_samples=100, a=1.0, b=0.0, sigma=0.2):
    """Generate linear data with random noise"""
    X = np.random.rand(n_samples, 1) * 2.0
    y = a * X + b + np.random.randn(n_samples, 1) * sigma
    return X, y



def generate_data_gauss(n_samples, means, sigmas, labels=None):
    """Generate random data from multiple Gaussian distributions"""
    n_classes = len(means)
    if labels is None:
        labels = np.arange(n_classes)
    else:
        assert len(labels) == n_classes, "Number of labels must match number of classes"
    X_list = []
    y_list = []
    for mean, sigma, label in zip(means, sigmas, labels):
        X = np.random.randn(n_samples, len(mean)) * sigma + mean
        y = np.full(n_samples, label)
        X_list.append(X)
        y_list.append(y)
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    indices = np.arange(n_samples * n_classes)
    np.random.shuffle(indices)

    return X[indices], y[indices]


def generate_data_outliers(n_samples, n_outliers, means, sigmas):
    """Generate random data with outliers"""
    X1 = np.random.randn(n_samples - n_outliers, 2) * sigmas[0] + means[0]
    Y1 = np.full(len(X1), 1)
    X2 = np.random.randn(n_samples - n_outliers, 2) * sigmas[1] + means[1]
    Y2 = np.full(len(X2), -1)
    X_out = np.random.randn(n_outliers, 2) * sigmas[2] + means[2]
    Y_out = np.full(len(X_out), 0)
    X = np.concatenate([X1, X2, X_out], axis=0)
    y = np.concatenate([Y1, Y2, Y_out], axis=0)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    return X[indices], y[indices]