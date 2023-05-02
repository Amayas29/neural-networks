import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def plot_2d(X, y="darkseagreen", title=""):
    plt.figure()
    plt.title(title)
    plt.scatter(X[:, 0], X[:, 1], s=40, c=y, label="data")
    plt.legend()
    plt.show()


def plot_net(
    X,
    y,
    net,
    train_loss,
    test_loss=None,
    loss_name="MSE",
    net_type="classif",
    data_xlabel="",
    data_ylabel="",
    net_title="",
):
    ncols = 2
    if net_type == "multiclass" or net_type == "auto_encodeur":
        ncols = 1

    fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(20, 6))

    if net_type == "multiclass" or net_type == "auto_encodeur":
        axs = [axs]

    axs[0].plot(train_loss, label=f"{loss_name} in Train", c="steelblue")

    if test_loss is not None and len(test_loss) != 0:
        axs[0].plot(test_loss, label=f"{loss_name} in Test", c="coral")

    axs[0].set_xlabel("Nombre d'itérations")
    axs[0].set_ylabel("Loss")
    axs[0].set_title(f"Evolution de la {loss_name}")
    axs[0].legend()

    if net_type == "multiclass" and net_type == "auto_encodeur":
        fig.suptitle(net_title)
        plt.show()
        return

    colors = ["darksalmon", "skyblue"]
    markers = ["o", "x"]

    classes = [-1, 1]
    if net.classes_type == "0/1":
        classes = [0, 1]

    if net_type == "reglin":
        X = np.column_stack((X, y))

    if net_type == "reglin":
        w = net.modules[0]._parameters["W"][0][0]
        toPlot = [w * x for x in X[:, 0]]

        axs[1].scatter(X[:, 0], X[:, 1], c="midnightblue", label="data")
        axs[1].set_xlabel(data_xlabel)
        axs[1].set_ylabel(data_ylabel)
        axs[1].plot(X[:, 0], toPlot, lw=4, color="r", label="reglin")
        axs[1].set_title("Droite de la régression")
        axs[1].legend()

    elif net_type == "classif":
        axs[1].set_title(f"Frontiere de décision pour {len(classes)} classes")

        y = y.reshape(-1)
        for i, cl in enumerate(classes):
            X_cl = X[y == cl]
            axs[1].scatter(
                X_cl[:, 0],
                X_cl[:, 1],
                c=colors[i],
                marker=markers[i],
                label=f"Classe : {cl}",
            )

        mmax = X.max(0)
        mmin = X.min(0)

        step = 1000
        x1grid, x2grid = np.meshgrid(
            np.linspace(mmin[0], mmax[0], step), np.linspace(mmin[1], mmax[1], step)
        )

        grid = np.hstack(
            (x1grid.reshape(x1grid.size, 1), x2grid.reshape(x2grid.size, 1))
        )

        res = net.predict(grid)
        res = res.reshape(x1grid.shape)

        axs[1].contourf(
            x1grid,
            x2grid,
            res,
            colors=colors,
            levels=[-1000, 0, 1000],
            alpha=0.4,
        )

        axs[1].set_xlabel(data_xlabel)
        axs[1].set_ylabel(data_ylabel)
        axs[1].legend()

    fig.suptitle(net_title)
    plt.show()


def visualization(X_train, Xhat, y_train, type_affichage="tsne", n_components=2):
    if type_affichage == "tsne":
        tsne = TSNE(n_components=n_components, random_state=0)
        aff_train = tsne.fit_transform(X_train)

        tsne = TSNE(n_components=n_components, random_state=0)
        aff_hat = tsne.fit_transform(Xhat)

    if type_affichage == "pca":
        tsne = PCA(n_components=n_components, random_state=0)
        aff_train = tsne.fit_transform(X_train)

        tsne = PCA(n_components=n_components, random_state=0)
        aff_hat = tsne.fit_transform(Xhat)

    if n_components == 2:
        fig = plt.figure(figsize=(12, 6))

        ax1 = fig.add_subplot(121)
        ax1.scatter(aff_train[:, 0], aff_train[:, 1], c=y_train)
        ax1.set_xlabel("Dimension 1")
        ax1.set_ylabel("Dimension 2")
        ax1.set_title(type_affichage.upper() + " Visualization for X_train")

        ax2 = fig.add_subplot(122)
        ax2.scatter(aff_hat[:, 0], aff_hat[:, 1], c=y_train)
        ax2.set_xlabel("Dimension 1")
        ax2.set_ylabel("Dimension 2")
        ax2.set_title(type_affichage.upper() + " Visualization for Xhat")

    if n_components == 3:
        fig = plt.figure(figsize=(12, 12))
        ax1 = fig.add_subplot(121, projection="3d")
        ax1.scatter(aff_train[:, 0], aff_train[:, 1], aff_train[:, 2], c=y_train)
        ax1.set_xlabel("Dimension 1")
        ax1.set_ylabel("Dimension 2")
        ax1.set_zlabel("Dimension 3")
        ax1.set_title(type_affichage.upper() + " Visualization in 3D for X_train")

        ax2 = fig.add_subplot(122, projection="3d")
        ax2.scatter(aff_hat[:, 0], aff_hat[:, 1], aff_hat[:, 2], c=y_train)
        ax2.set_xlabel("Dimension 1")
        ax2.set_ylabel("Dimension 2")
        ax2.set_zlabel("Dimension 3")
        ax2.set_title(type_affichage.upper() + " Visualization in 3D for Xhat")

    plt.tight_layout()
    plt.show()


def plot_usps_predictions(X, Xhat, indices):
    num_images = len(indices)
    fig, axs = plt.subplots(nrows=2, ncols=num_images, figsize=(16, 6))
    for i, idx in enumerate(indices):
        axs[0, i].imshow(X[idx].reshape((16, 16)))
        axs[0, i].set_title(f"Image originale {idx}")
        axs[0, i].axis("off")
        axs[1, i].imshow(Xhat[idx].reshape((16, 16)))
        axs[1, i].set_title(f"Image reconstruite {idx}")
        axs[1, i].axis("off")
    fig.tight_layout()
    plt.show()
