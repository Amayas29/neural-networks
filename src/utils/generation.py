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


def generate_echiquier(n_samples, sigmas, n = 4):

    means = [[i, j] for i in range(1,n+1) for j in range(1,n+1)]
    sig = [np.eye(2) * sigmas] * len(means)
    labels = [0 if i%2 == j%2 else 1 for i in range(1,n+1) for j in range(1,n+1)]

    # Génération des données
    return generate_data_gauss(n_samples, means, sig, labels)

def generate_data_sphere(n_samples, locs, classes):
    X_list = []
    y_list = []
    
    for cl, loc in zip(classes, locs):
        theta = np.linspace(0, 2*np.pi, n_samples)
        r = np.random.normal(loc=loc, scale=0.6, size=(n_samples,)) 
        x = r * np.cos(theta)
        y = r * np.sin(theta)

        X_list.append(np.vstack((x, y)).T)
        y_list.append([cl] * len(y))
        
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    index = np.arange(len(X))
    np.random.shuffle(index)

    return X[index], y[index]