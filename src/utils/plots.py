import numpy as np
import matplotlib.pyplot as plt


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
    loss_name="MSE",
    net_type="classif",
    data_xlabel="",
    data_title="Data",
    net_title="",
):
    ncols = 2
    if net_type == "multiclass":
        ncols = 1

    fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(20, 6))

    axs[0].plot(train_loss, label=loss_name, c="darkseagreen")
    axs[0].set_xlabel("Nombre d'itérations")
    axs[0].set_title(f"Evolution de la {loss_name}")
    axs[0].legend()

    colors = ["darksalmon", "skyblue"]
    markers = ["o", "x"]
    classes = net.classes

    if net_type == "reglin":
        X = np.column_stack((X, y))

    if net_type == "reglin":
        w = net.modules[0]._parameters["W"][0][0]
        toPlot = [w * x for x in X[:, 0]]

        axs[1].scatter(X[:, 0], X[:, 1], c="midnightblue", label="data")
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

        axs[1].legend()

    fig.suptitle(net_title)
    plt.show()
