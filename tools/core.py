import numpy as np
import helpers

class PCA(object):
    def __init__(self):
        self.eigenvectors = None
        self.eigenvalues = None
        self.feature_means = None

    def fit(self, data):
        """
        Computes and stores the principal components of the dataset.

        :param data: 2D numpy array (rows = samples, columns = features)
        """
        self.feature_means = np.mean(data, axis=0)
        centered = data - self.feature_means

        # Compute covariance matrix
        cov = np.cov(centered, rowvar=False)

        # Eigen decomposition
        self.eigenvalues, self.eigenvectors = np.linalg.eig(cov)

        # Sort in descending order
        sorted_order = np.argsort(self.eigenvalues)[::-1]
        self.eigenvalues = self.eigenvalues[sorted_order]
        self.eigenvectors = self.eigenvectors[:, sorted_order]

    def transform(self, data, n_components=1):
        """
        Rotates the given data into the principal components space.

        :param data: 2D numpy array
        :param n_components: Number of principal components to retain
        :return: Transformed 2D numpy array
        """
        if self.eigenvectors is None or self.feature_means is None:
            raise ValueError("PCA model has not been fitted yet. Call fit() first.")

        return np.dot(data - self.feature_means, self.eigenvectors[:, :n_components])


class Cluster(object):
    def __init__(self, k):
        self.k = k
        self.centroids = None

    def fit(self, data, iterations=10):
        """
        Runs K-Means clustering on the dataset.

        :param data: 2D numpy array (rows = samples, columns = features)
        :param iterations: Number of iterations to refine cluster assignments
        """
        n_dims = data.shape[1]  # Number of features, not samples

        # Initialize centroids randomly within the feature range
        self.centroids = [helpers.random_vector_range(np.amin(data, axis=0),
                                                      np.amax(data, axis=0),
                                                      n_dims) for _ in range(self.k)]

        for _ in range(iterations):
            clusters = [[] for _ in range(self.k)]  # Reset clusters each iteration

            # Assign each data point to the closest centroid
            for vec in data:
                closest_centroid = np.argmin([helpers.vec_distance(c, vec) for c in self.centroids])
                clusters[closest_centroid].append(vec)

            # Update centroids
            for i in range(len(self.centroids)):
                if clusters[i]:  # Avoid empty clusters
                    self.centroids[i] = np.mean(clusters[i], axis=0)
