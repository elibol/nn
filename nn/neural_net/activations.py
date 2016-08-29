import autograd.numpy as np
from autograd.scipy.misc import logsumexp

# leaky rlu (rectified linear unit)
# batch normalization

# Add additional targets:
# What was prescribed
# What we think is best
# What the model prescribes

def sigmoid(X):
    return 0.5 * (np.tanh(X * 0.5) + 1.0)
    # return 1.0/(1.0 + np.exp(-X))


def tanh(x):
    return np.tanh(x)


def relu(X):
    return np.clip(X, 0, np.finfo(X.dtype).max)


def softmax(X):
    return X - logsumexp(X, axis=1, keepdims=True)
    # tmp = X - X.max(axis=1)[:, np.newaxis]
    # X = np.exp(tmp)
    # X /= X.sum(axis=1)[:, np.newaxis]
    # return X


def identity(X):
    return X
