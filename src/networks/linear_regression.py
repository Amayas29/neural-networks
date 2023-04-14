from loss.mse_loss import MSELoss
from module.linear import Linear
import numpy as np


class LinearRegression:

    def __init__(self, niter=1000, gradient_step=1e-3, loss=MSELoss()):
        self.niter = niter
        self.gradient_step = gradient_step

        self.loss = loss
        self.linear = None
        self.train_loss = []

    def fit(self, X, y):

        bX, input_dim = X.shape
        bY, output_dim = y.shape

        assert bX == bY, "Erreur - LinearRegression:fit - Les dimensions batch de X et y ne correspondent pas."

        self.linear = Linear(input_dim, output_dim)
        self.train_loss = []

        for _ in range(self.niter):

            yhat = self.linear.forward(X)
            loss_i = self.loss.forward(y, yhat)
            self.train_loss.append(np.mean(loss_i))

            delta = self.loss.backward(y, yhat)
            self.linear.backward_update_gradient(X, delta)

            self.linear.update_parameters(self.gradient_step)
            self.linear.zero_grad()

    def predict(self, X):
        return self.linear.forward(X)
