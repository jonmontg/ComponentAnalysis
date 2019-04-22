import numpy as np


def covariance_matrix(data):
    """
    Returns the covariance matrix of the data

    :param data: 2d numpy array where each column is a feature and each row is a data point
    :return: 2d square numpy array
    """
    return np.divide(np.matmul(np.transpose(data), data), len(data))
