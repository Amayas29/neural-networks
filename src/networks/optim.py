from loss.mse_loss import MSELoss
import numpy as np


class Optim:

    def __init__(self, net, loss=MSELoss(), eps=1e-5):
        self.net = net
        self.loss = loss
        self.eps = eps
        self.train_loss = []

    def step(self, X, y):
        yhat = self.net.forward(X)
        delta = self.loss.backward(y, yhat)
        self.net.backward(delta, eps=self.eps)

    def SGD(self, X, y, batch_size, num_iterations, verbose=True):

        if len(X.shape) == 1:
            X = X.reshape((-1, 1))

        if len(y.shape) == 1:
            y = y.reshape((-1, 1))

        self.train_loss = []

        num_samples = X.shape[0]
        num_batches = num_samples // batch_size

        for iteration in range(num_iterations):

            indices = np.random.permutation(num_samples)
            X = X[indices]
            y = y[indices]

            for batch in range(num_batches):
                start = batch * batch_size
                end = (batch + 1) * batch_size
                X_batch = X[start:end]
                y_batch = y[start:end]

                self.step(X_batch, y_batch)

            y_pred = self.net.forward(X)
            loss = self.loss.forward(y, y_pred).mean()
            self.train_loss.append(loss)

            if verbose and (iteration+1) % 100 == 0:
                print(
                    f"Iteration {iteration+1}/{num_iterations} - Loss: {loss}")

        print("Training completed.")
