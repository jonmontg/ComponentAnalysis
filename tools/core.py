import numpy as np
import tools.helpers as helpers


class PCA(object):

    def __init__(self):
        self.eigenvectors = None
        self.eigenvalues = None
        self.feature_means = None

    def fit(self, data):
        """
        Returns the first n principal components of the data

        :param data: 2d numpy array - each column is one feature and each row is one data
        :param n: int - number of principal components
        :return:
        """
        # subtract off the mean of each column to center the data
        self.feature_means = np.mean(data, axis=0)
        centered = np.subtract(data, self.feature_means)

        # find the covariance matrix of the centered dataa
        cov = helpers.covariance_matrix(centered)

        # find the eigenvalues and eigenvectors of the covariance matrix
        self.eigenvalues, self.eigenvectors = np.linalg.eig(cov)

        # sort the eigenvalues and eigenvectors in decreasing order of eigenvalues
        sorted_order = np.argsort(self.eigenvalues)[::-1]
        self.eigenvalues = self.eigenvalues[sorted_order]
        self.eigenvectors = self.eigenvectors[:, sorted_order]

    def transform(self, data, n_components=1):
        """
        Rotates the given data into the principal components space

        :param data: 2d numpy array
        :param n_components: dimensionality of the transformed data
        :return: 2d numpy array
        """
        return np.dot(data-self.feature_means, self.eigenvectors[:, :n_components])

