from tools.helpers import covariance_matrix
import matplotlib.pyplot as plt
from tools.core import PCA as my_PCA
from sklearn.decomposition import PCA as benchmark_PCA
from sklearn.datasets import load_iris
import numpy as np
from time import time


def sklearn_example_test():
    print("sklearn sample data PCA test")
    data = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])

    bPCA = benchmark_PCA(n_components=1)
    myPCA = my_PCA()

    bPCA.fit(data)
    myPCA.fit(data)

    assert np.array_equal(myPCA.transform(data, 1), bPCA.transform(data)) or \
           np.array_equal(myPCA.transform(data, 1), - bPCA.transform(data))
    bPCA = benchmark_PCA(n_components=2)
    bPCA.fit(data)
    assert np.array_equal(myPCA.transform(data, 2), bPCA.transform(data)) or \
           np.array_equal(myPCA.transform(data, 2), - bPCA.transform(data))
    print("Test Passed")

    print("Iris data PCA test")
    iris = load_iris()
    data = iris.data
    bPCA = benchmark_PCA(n_components=2)
    start_t = time()
    bPCA.fit(data)
    print("Benchmark fit time: %f" % (time() - start_t))
    start_t = time()
    myPCA.fit(data)
    print("My fit time:        %f" % (time() - start_t))

    rotdata = np.transpose(myPCA.transform(data, 2))
    plt.scatter(rotdata[0], -rotdata[1], c='b')
    rotdata = np.transpose(bPCA.transform(data))
    plt.scatter(rotdata[0], rotdata[1], c='r')
    plt.show()
    plt.clf()


if __name__=="__main__":
    sklearn_example_test()
