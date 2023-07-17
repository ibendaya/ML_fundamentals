import numpy as np


def unit_step_func(x):
    return np.where(x > 0, 1, 0)


class Perceptron:
    def __init__(self, lr=0.01, iters=1000):
        self.lr = lr
        self.iters = iters
        self.activation_function = unit_step_func
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # init params
        self.weights = np.zeros(n_features)  # would be better with random inits
        self.bias = 0

        y_ = np.where(y > 0, 1, 0)

        # learn weights

        for _ in range(self.iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_pred = self.activation_function(linear_output)

                # update weight and bias
                update = self.lr * (y_[idx] - y_pred)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_pred = self.activation_function(linear_output)

        return y_pred
