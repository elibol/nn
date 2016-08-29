import numpy as np


def multilabel_data(num_rows=10,
                    num_feats=3,
                    num_labels=5,

                    num_nonzero_X=10,
                    nonzero_low=1,
                    nonzero_high=10,

                    seed=1337,
                    hidden_layer_sizes=None,
                    target_dtype=np.bool,
                    dist=None,):

    if dist is None:
        dist = (np.random.normal, {"loc": 0, "scale": 1})

    dist_func, dist_params = dist

    if hidden_layer_sizes is None:
        hidden_layer_sizes = []
    np.random.seed(seed)
    X = np.zeros(shape=(num_rows, num_feats))

    row_idx = np.sort(np.random.randint(0, num_rows, num_nonzero_X))
    col_idx = np.random.randint(0, num_feats, num_nonzero_X)
    vals = np.random.randint(nonzero_low, nonzero_high, num_nonzero_X)
    X[row_idx, col_idx] = vals

    W = map(lambda nm: dist_func(size=nm[0]*nm[1], **dist_params).reshape(nm), (lambda x: zip(*(x[:-1], x[1:])))([num_feats] + hidden_layer_sizes + [num_labels]))

    res = X
    for layer in W:
        res = np.dot(res, layer)
    Y = res > np.mean(res)

    # Y = np.zeros(shape=(num_rows, num_labels), dtype=target_dtype)
    # row_idx = np.sort(np.random.randint(0, num_rows, num_nonzero_Y))
    # col_idx = np.random.randint(0, num_labels, num_nonzero_Y)
    # vals = np.ones(num_nonzero_Y, dtype=target_dtype)
    # Y[row_idx, col_idx] = vals

    X_mask = np.all(np.equal(X, 0), axis=1)
    Y_mask = np.all(np.equal(Y, 0), axis=1)
    Y_mask |= np.all(np.equal(Y, 1), axis=1)
    mask = X_mask | Y_mask
    X = X[~mask]
    Y = Y[~mask]

    return X, Y
