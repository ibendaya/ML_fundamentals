import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class LogisticRegression:
    def __init__(self, lr=0.001, n_iter=1000):
        self.lr = lr
        self.n_iter = n_iter
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iter):
            linear_predictions = (
                np.dot(X, self.weights) + self.bias
            )  # (n_smaples, n_features) @ (n_features, 1) --> (n_samples, 1)
            predictions = sigmoid(linear_predictions)

            dw = (1 / n_samples) * np.dot(
                X.T, (predictions - y)
            )  # (n_sample, n_features)T @ (n_features, 1)--> (n_features, 1)
            db = (1 / n_samples) * np.sum(predictions - y)  # (1, 1)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        linear_predictions = (
            np.dot(X, self.weights) + self.bias
        )  # (n_smaples, n_features) @ (n_features, 1) --> (n_samples, 1)
        y_predictions = sigmoid(linear_predictions)

        class_predictions = [0 if y < 0.5 else 1 for y in y_predictions]

        return class_predictions
