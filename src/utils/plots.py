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
    optim,
    X,
    y,
    net_type="classif",
    net_title="",
    data_xlabel="",
    data_ylabel="",
    display_loss=True,
    display_boundary=True,
    display_score=True,
):
    if net_type == "reglin":
        X = np.column_stack((X, y))
        display_score = False

    elif net_type == "multiclass":
        display_boundary = False

    elif net_type == "auto_encodeur":
        display_boundary = False
        display_score = False

    ncols = np.array([display_loss, display_score, display_boundary]).sum()

    if ncols == 0:
        return

    fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(20, 6))

    if ncols == 1:
        axs = [axs]

    i = 0

    if display_loss:
        loss_name = optim.loss.__class__.__name__

        axs[i].plot(optim.train_loss, label=f"{loss_name} in Train", c="steelblue")

        if optim.test_loss is not None and len(optim.test_loss) != 0:
            axs[i].plot(optim.test_loss, label=f"{loss_name} in Test", c="coral")

        axs[i].set_xlabel("Nombre d'itérations")
        axs[i].set_ylabel("Loss")
        axs[i].set_title(f"Evolution de la {loss_name}")
        axs[i].legend()

        i += 1

    if display_boundary:
        if net_type == "reglin":
            w = optim.net.modules[0]._parameters["W"][0][0]
            toPlot = [w * x for x in X[:, 0]]

            axs[i].scatter(X[:, 0], X[:, 1], c="midnightblue", label="data")
            axs[i].set_xlabel(data_xlabel)
            axs[i].set_ylabel(data_ylabel)
            axs[i].plot(X[:, 0], toPlot, lw=4, color="r", label="reglin")
            axs[i].set_title(f"Droite de la régression avec â = {w:.2f}")
            axs[i].legend()

        elif net_type == "classif":
            colors = ["darksalmon", "skyblue"]
            markers = ["o", "x"]

            classes = [-1, 1]
            if optim.net.classes_type == "0/1":
                classes = [0, 1]

            axs[i].set_title(f"Frontiere de décision pour {len(classes)} classes")

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

            res = optim.net.predict(grid)
            res = res.reshape(x1grid.shape)

            axs[i].contourf(
                x1grid,
                x2grid,
                res,
                colors=colors,
                levels=[-1000, 0, 1000],
                alpha=0.4,
            )

            axs[i].set_xlabel(data_xlabel)
            axs[i].set_ylabel(data_ylabel)
            axs[i].legend()

        i += 1

    if display_score:
        axs[i].plot(optim.train_score, label="score in Train", c="steelblue")

        if optim.test_score is not None and len(optim.test_score) != 0:
            axs[i].plot(optim.test_score, label="score in Test", c="coral")

        axs[i].set_ylim(0, 1.1)
        axs[i].set_xlabel("Nombre d'itérations")
        axs[i].set_ylabel("Score")
        axs[i].set_title("Evolution du score")
        axs[i].legend()

        i += 1

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
    figsize = (4 * num_images, 6)
    fig, axs = plt.subplots(nrows=2, ncols=num_images, figsize=figsize)

    for i, idx in enumerate(indices):
        axs[0, i].imshow(X[idx].reshape((16, 16)))
        axs[0, i].set_title(f"Image originale {idx}")
        axs[0, i].axis("off")
        axs[1, i].imshow(Xhat[idx].reshape((16, 16)))
        axs[1, i].set_title(f"Image reconstruite {idx}")
        axs[1, i].axis("off")
    fig.tight_layout()
    plt.show()
