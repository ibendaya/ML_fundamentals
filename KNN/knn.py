import numpy as np


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
        # calc euclidian distance
        distances = [euc_distance(x, x_train) for x_train in self.X_train]

        # sort
        idx_k_closest = np.argsort(distances)[0 : self.k]

        # vote
        hashmap = {}  # key: label, val: count of label
        for idx in idx_k_closest:
            hashmap[self.y_train[idx]] = 1 + hashmap.get(self.y_train[idx], 0)

        key_list = list(hashmap.keys())
        val_list = list(hashmap.values())

        most_common_label = key_list[val_list.index(max(val_list))]

        return most_common_label
