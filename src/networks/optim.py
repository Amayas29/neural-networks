from loss.loss import MSELoss
import numpy as np
from tqdm import tqdm

from sklearn.model_selection import train_test_split


class Optim:
    def __init__(self, net, loss=MSELoss(), eps=1e-5):
        self.net = net
        self.loss = loss
        self.eps = eps
        self.train_loss = []
        self.test_loss = []

        self.train_score = []
        self.test_score = []

    def step(self, X, y):
        yhat = self.net(X)

        delta = self.loss.backward(y, yhat)
        self.net.zero_grad()
        self.net.backward(delta)
        self.net.update_parameters(self.eps)

    def SGD(self, X, y, batch_size, epochs, test_train_split=False, verbose=True):
        if X.ndim == 1:
            X = X.reshape((-1, 1))

        if y.ndim == 1:
            y = y.reshape((-1, 1))

        if test_train_split:
            X, X_test, y, y_test = train_test_split(X, y, test_size=0.2)

        if batch_size > len(X):
            batch_size = len(X)

        self.train_loss = []
        self.test_loss = []

        self.train_score = []
        self.test_score = []

        n_samples = X.shape[0]
        n_batches = n_samples // batch_size

        for epoch in tqdm(range(epochs)):
            loss_epoch = 0

            indices = np.random.permutation(n_samples)
            X = X[indices]
            y = y[indices]

            X_batchs = np.array_split(X, n_batches)
            y_batchs = np.array_split(y, n_batches)

            for X_batch, y_batch in zip(X_batchs, y_batchs):
                self.step(X_batch, y_batch)

            loss_epoch /= n_batches

            yhat = self.net(X)
            loss_epoch = self.loss(y, yhat).mean()
            self.train_loss.append(loss_epoch)

            score_epoch = self.net.score(X, y)
            self.train_score.append(score_epoch)

            if test_train_split:
                y_test_hat = self.net(X_test)
                loss_value = self.loss(y_test, y_test_hat).mean()
                self.test_loss.append(loss_value)

                score_epoch = self.net.score(X_test, y_test)
                self.test_score.append(score_epoch)

            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {loss_epoch}")

        print("Training completed.")
