import numpy as np

from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler


def standardize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[np.where(std == 0.0)[0]] = 1.0

    return (X - mean) / std


def standardize_sparse(X, axis=0):
    ss = StandardScaler(with_mean=False)
    return ss.fit_transform(X)

    N = col.shape[axis]
    mean = csr_matrix.mean(X, axis=axis)
    sqr = X.copy() # take a copy of the col
    sqr.data **= 2 # square the data, i.e. just the non-zero data
    variance = sqr.sum()/N - mean**2
    std = np.sqrt(variance)
    std[np.where(std == 0.0)[0]] = 1.0
    
    return (X - mean) / std


def corr(x, y):
    return np.sum(x*y) / y.shape[0]
