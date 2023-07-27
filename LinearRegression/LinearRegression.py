import numpy as np


class LinearRegression:
    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            y_pred = (
                np.dot(X, self.weights) + self.bias
            )  # (n_sample, n_features) @ (n_features, 1) ---> (n_samples, 1)

            dw = (1 / n_samples) * np.dot(
                X.T, (y_pred - y)
            )  # (n_features, n_samples) @ (n_samples, 1) ---> (n_features, 1)

            db = (1 / n_samples) * sum(y_pred - y)  # (1, 1)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        y_est = (
            np.dot(X, self.weights) + self.bias
        )  # (n_sample, n_features) @ (n_features, 1) ---> (n_samples, 1)

        return y_est
