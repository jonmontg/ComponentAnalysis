import numpy as np


def covariance_matrix(data):
    """
    Returns the covariance matrix of the data

    :param data: 2d numpy array where each column is a feature and each row is a data point
    :return: 2d square numpy array
    """
    return np.divide(np.dot(np.transpose(data), data), len(data))


def mean_vector(data):
    """
    Returns the vector at the mean location in cartesian space

    :param data: 2d array. Rows are vectors, columns are dimensions
    :return: numpy array
    """
    return np.mean(data, axis=0)


def random_vector_range(minvec, maxvec, n_dims):
    r_vec = np.zeros_like(minvec)
    for i in range(n_dims):
        r_vec[i] = np.random.uniform(minvec[i], maxvec[i])
    return r_vec


def vec_distance(a, b):
    """
    Returns the euclidean distance between two vectors
    :param a: 1d numpy array
    :param b: 1d numpy array
    :return: float
    """
    assert len(a) == len(b)
    return np.sqrt(np.sum(np.square(np.subtract(a, b))))
