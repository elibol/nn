import unittest
import collections
import numpy as np


class NumpyTestCase(unittest.TestCase):

    def test_wipe_and_iterate(self):

        def check_ones():
            for i in range(X.shape[0]):
                if i in row_idx:
                    indexes = np.where(i == row_idx)[0]
                    # given i in row_idx, if j in col_idx[indexes] then X[i][j] should be 1
                    if j in col_idx[indexes]:
                        assert X[i][j] == 1.0
                    else:
                        assert X[i][j] == 0.0
                else:
                    for j in range(X.shape[1]):
                        assert X[i][j] == 0.0

        def check_all_zero():
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    assert X[i][j] == 0

        num_rows = 1000
        num_feats = 10
        X_zeros = np.zeros(shape=(num_rows, num_feats))
        X = np.zeros(shape=(num_rows, num_feats))

        assert np.allclose(X_zeros, X)

        N = 100
        row_idx = np.sort(np.random.randint(0, num_rows, N))
        col_idx = np.random.randint(0, num_feats, N)
        vals = np.ones(N)

        X[row_idx, col_idx] = vals
        check_ones()

        X[:] = 0
        check_all_zero()

        X[[1,1,2,2],[1,1,2,2]] = [1,1,1,1]
        assert np.sum(X) == 2
        assert X.shape == X_zeros.shape

    def test_ordered_dict(self):
        start = 10
        end = 20
        id_range = range(start, end)
        patient_ids = map(str, id_range)
        data = []
        for patient_id in patient_ids:
            patient_data = range(int(patient_id))
            for datum in patient_data:
                data.append((patient_id, datum))

        data_dict = collections.OrderedDict()
        for item in data:
            patient_id, datum = item
            if patient_id not in data_dict:
                data_dict[patient_id] = []
            data_dict[patient_id].append(datum)

        # make sure it retained order
        data_keys = data_dict.keys()
        for i in range(len(data_keys)):
            assert data_keys[i] == patient_ids[i]
        for i, val in enumerate(data_dict.values()):
            assert len(val) == start + i
