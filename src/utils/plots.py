import numpy as np
import matplotlib.pyplot as plt


def plot_frontiere_lineaire(X, y, nets, step=30):
    mmax = X.max(0)
    mmin = X.min(0)

    x1grid, x2grid = np.meshgrid(np.linspace(
        mmin[0], mmax[0], step), np.linspace(mmin[1], mmax[1], step))

    grid = np.hstack((x1grid.reshape(x1grid.size, 1),
                     x2grid.reshape(x2grid.size, 1)))

    fig, axs = plt.subplots(nrows=1, ncols=len(nets), figsize=(6, 6))

    for i, net in enumerate(nets):

        res = net.forward(grid)
        res = res.reshape(x1grid.shape)

        axs[i].set_title(f"Frontiere de décision pour {len(classes)} classes")

        classes = np.unique(y)

        for cl in classes:
            X_cl = X[y == cl]
            axs[i].scatter(X_cl[:, 0], X_cl[:, 1],
                           s=20, label=f'Classe : {cl}')

        axs[i].contourf(x1grid, x2grid, res, colors=[
            "darksalmon", "skyblue"], levels=[-1000, 0, 1000], alpha=0.5)

        axs[i].legend()

    plt.tight_layout()
    plt.show()

    # for net in nets:
    #     res = net.forward(grid)
    #     res = res.reshape(x1grid.shape)

    #     plt.figure()
    #     classes = np.unique(y)
    #     for cl in classes:
    #         X_cl = X[y == cl]
    #         plt.scatter(X_cl[:, 0], X_cl[:, 1], s=20, label=f'Classe : {cl}')

    #     plt.contourf(x1grid, x2grid, res, colors=[
    #         "darksalmon", "skyblue"], levels=[-1000, 0, 1000], alpha=0.5)

    #     plt.title(f"Frontiere de décision pour {len(classes)} classes")
    #     plt.legend()
    #     plt.show()


def plot_frontiere_reglin(X, weights):
    fig, axs = plt.subplots(nrows=1, ncols=len(weights), figsize=(20, 5))

    for i, w in enumerate(weights):
        toPlot = [w * x for x in X[:, 0]]

        axs[i].scatter(X[:, 0], X[:, 1], s=20, c='midnightblue', label='data')
        axs[i].plot(X[:, 0], toPlot, lw=4, color='r', label='reglin')
        axs[i].set_title('w={:.2f}'.format(w))
        axs[i].legend()

    fig.suptitle("Regression lineaire")
    plt.show()


def plot_2d(X, title):
    plt.figure()
    plt.title(title)
    plt.scatter(X[:, 0], X[:, 1], s=20, c='darkseagreen', label='data')
    plt.legend()
    plt.show()


def plot_loss(net, loss_name):
    print(f"Last Loss : {net.train_loss[-1]}")

    plt.figure()
    plt.title(f'Evolution de la {loss_name}')
    plt.plot(net.train_loss, label=loss_name, c='darkseagreen')
    plt.xlabel('Nombre d\'itérations')
    plt.legend()
    plt.show()
