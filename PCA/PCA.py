import numpy as np


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean = None
        self.components = None

    def fit(self, X):
        # mean centering
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        # covariance
        cov = np.cov(X.T)  # samples as columns so T

        # eigenvectors, eigen values
        eigen_vectors, eigen_values = np.linalg.eig(cov)
        eigen_vectors = eigen_vectors.T

        # sort by eigenvalues
        idxs = np.argsort(eigen_values)[::-1]  # decreasing order [::-1]
        eigen_values = eigen_values[idxs]
        eigen_vectors = eigen_vectors[idxs]

        self.components = eigen_vectors[: self.n_components]

    def transform(self, X):
        # project the data
        X = X - self.mean
        return np.dot(X, self.components.T)
