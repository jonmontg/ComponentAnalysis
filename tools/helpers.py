import numpy as np


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
