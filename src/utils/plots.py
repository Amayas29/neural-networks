import numpy as np
import matplotlib.pyplot as plt


def plot_2d(X, y='darkseagreen', title=""):
    plt.figure()
    plt.title(title)
    plt.scatter(X[:, 0], X[:, 1], s=20, c=y, label='data')
    plt.legend()
    plt.show()


def plot_net(X, y, net, train_loss, loss_name="MSE", net_type="classif", data_xlabel="", data_title="Data", net_title=""):

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

    if net_type == "reglin":
        X = np.column_stack((X, y))

    color = "darkseagreen"
    if net_type == "classif":
        color = y

    axs[0].scatter(X[:, 0], X[:, 1], s=20, c=color, label='data')
    axs[0].set_xlabel(data_xlabel)
    axs[0].set_title(data_title)
    axs[0].legend()

    axs[1].plot(train_loss, label=loss_name, c='darkseagreen')
    axs[1].set_xlabel('Nombre d\'itérations')
    axs[1].set_title(f"Evolution de la {loss_name}")
    axs[1].legend()

    if net_type == "reglin":
        w = net.modules[0]._parameters[0][0]
        toPlot = [w * x for x in X[:, 0]]

        axs[2].scatter(X[:, 0], X[:, 1], s=20, c='midnightblue', label='data')
        axs[2].plot(X[:, 0], toPlot, lw=4, color='r', label='reglin')
        axs[2].set_title("Droite de la régression")
        axs[2].legend()

    elif net_type == "classif":
        mmax = X.max(0)
        mmin = X.min(0)

        step = 40
        x1grid, x2grid = np.meshgrid(np.linspace(
            mmin[0], mmax[0], step), np.linspace(mmin[1], mmax[1], step))

        grid = np.hstack((x1grid.reshape(x1grid.size, 1),
                          x2grid.reshape(x2grid.size, 1)))

        res = net.forward(grid)
        res = res.reshape(x1grid.shape)

        classes = np.unique(y)
        axs[2].set_title(f"Frontiere de décision pour {len(classes)} classes")

        y = y.reshape(-1)
        for cl in classes:
            X_cl = X[y == cl]
            axs[2].scatter(X_cl[:, 0], X_cl[:, 1],
                           s=20, label=f'Classe : {cl}')

        axs[2].contourf(x1grid, x2grid, res, colors=[
            "darksalmon", "skyblue"], levels=[-1000, 0, 1000], alpha=0.5)

        axs[2].legend()

    fig.suptitle(net_title)
    plt.show()
