import numpy as np


def standardize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[np.where(std == 0.0)[0]] = 1.0
    return (X - mean) / std


def SVD(X):
    X -= np.mean(X, axis = 0) # zero-center the data (important)
    cov = np.dot(X.T, X) / X.shape[0] # get the data covariance matrix
    U, S, V = np.linalg.svd(cov)
    return U, S, V


def PCA(X, num_retained=1000):
    U,S,V = SVD(X)
    Xrot_reduced = np.dot(X, U[:,:100]) # Xrot_reduced becomes [N x 100]
    return Xrot_reduced


def whiten(X, num_retained=1000):
    U,S,V = SVD(X)
    Xrot = np.dot(X, U) # decorrelate the data
    Xwhite = Xrot / np.sqrt(S + 1e-5)
    return Xwhite
