import numpy as np
from collections import Counter


def euc_distance(x1, x2):
    return np.sqrt(sum((x1 - x2) ** 2))


class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self, x):
        # calculate distances
        distances = [euc_distance(x, x_train) for x_train in self.X_train]

        # sort and take k nearsest sample
        k_idx = np.argsort(distances)[0 : self.k]

        # vote
        counts = Counter(self.y[k_idx])
        most_common = counts.most_common(1)[0][0]

        return most_common
