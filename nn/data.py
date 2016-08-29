import os
from . import utils

from sklearn.cross_validation import train_test_split
import numpy as np
from scipy.sparse import csr_matrix


class AbstractReader(object):

    location = None
    reader = None

    def __init__(self, location):
        pass

    def get_reader(self):
        raise NotImplemented()

    def decode(self, data):
        raise NotImplemented()

    def read(self):
        raise NotImplemented()

    def next(self):
        data = self.read()
        data = self.decode(data)
        return data


class Data(object):

    def __init__(self, X, Y, seed=0, test_frac=0.1, ho_frac=0.2, sparse=False,
                 num_features=None, num_labels=None, batch_size=None):
        self.X = X
        self.Y = Y

        if num_features is None:
            self.num_features = self.X.shape[1]
        else:
            self.num_features = num_features

        if num_labels is None:
            self.num_labels = self.Y.shape[1]
        else:
            self.num_labels = num_labels

        self.seed = seed
        self.test_frac = test_frac
        self.ho_frac = ho_frac
        self.X_train, self.X_test, self.X_ho = None, None, None
        self.Y_train, self.Y_test, self.Y_ho = None, None, None

        if sparse:
            self.X = csr_matrix(X)
            self.Y = csr_matrix(Y)

        self.split()
        self.info = self.get_info()

        self.batch_size = batch_size
        if self.batch_size is None:
            self.batch_size = self.X_train[0]

    def get_info(self):
        info = {}
        info["num_rows_train"] = self.X_train.shape[0]
        info["num_rows_test"] = self.X_test.shape[0]
        info["num_rows_ho"] = self.X_ho.shape[0]
        return info

    def split(self):
        idx = np.arange(self.X.shape[0])
        train_test_idx, ho_idx = train_test_split(idx, test_size=self.ho_frac, random_state=self.seed)
        # rescale test frac given what's left
        test_frac = self.test_frac / (1-self.ho_frac)

        train_idx, test_idx = train_test_split(train_test_idx, test_size=test_frac, random_state=self.seed)

        assert train_idx.shape[0] + test_idx.shape[0] + ho_idx.shape[0] == idx.shape[0]

        self.X_train, self.X_test, self.X_ho = self.X[train_idx], self.X[test_idx], self.X[ho_idx]
        self.Y_train, self.Y_test, self.Y_ho = self.Y[train_idx], self.Y[test_idx], self.Y[ho_idx]

    def reset(self):
        pass

    def __iter__(self):
        return self

    def next(self):
        return self.X_train, self.Y_train


# TODO: Refactor / generalize
class SparseBatchIterator(object):
    def __init__(self,
                 base_dir,
                 filename="sparse_data.csv",
                 x_data_type=np.float32,
                 y_data_type=np.float32,
                 binarize=False,
                 batch_size=1000,
                 max_patients=None):

        self.ff_paths = config.FeatureFilePaths(base_dir=base_dir)

        self.batch_size = batch_size
        self.binarize = binarize
        self.max_patients = max_patients

        self.concepts_obj = Concepts()
        self.classes = sorted(self.concepts_obj.concept_dict_filtered.keys())
        self.classes = {cls: i for i, cls in enumerate(self.classes)}
        self.num_classes = len(self.classes.keys())

        self.x_data_type = x_data_type
        self.y_data_type = y_data_type
        self.base_dir = base_dir

        print "reading info"
        self.info = utils.read_json(self.ff_paths.info_filename)
        self.num_features = self.info['num_features']
        self.num_rows = self.info['num_rows']

        self.sparse_data_filename = os.path.join(self.base_dir, filename)

        print "reading rewards"
        self.reward_dict = utils.read_json(self.ff_paths.rewards_filename)

        self.iterable = None

        self.patient_ids = None
        self.X = np.zeros(shape=(batch_size, self.num_features), dtype=self.x_data_type)
        self.Y = np.zeros(shape=(batch_size, self.num_classes), dtype=self.y_data_type)
        self.Y_clinician = np.zeros(shape=(batch_size, self.num_classes), dtype=y_data_type)

    def start(self):
        self.iterable = utils.CSVIterator(self.sparse_data_filename, " ", 0, None, has_header=False)
        self.patient_count = 0

    def __iter__(self):
        return self

    def next(self):
        if self.max_patients is not None and self.patient_count >= self.max_patients:
            raise StopIteration

        self.X.fill(0)
        self.Y.fill(-1)
        self.Y_clinician.fill(-1)

        patient_ids = []
        row_index = 0

        y_clinician_row_idx = []
        y_clinician_col_idx = []

        y_col_idx = []
        y_row_idx = []

        x_col_idx = []
        x_row_idx = []
        x_val = []

        for entry in self.iterable:
            patient_id, y_clinician, y, x = entry[0], entry[1], entry[2], entry[3:]

            # patient_id
            patient_ids.append(patient_id)

            # TODO: collect clinician data
            y_clinician_col = [self.classes[cls] for cls in y_clinician.split("|")]
            y_clinician_row_idx += [row_index] * len(y_clinician_col)
            y_clinician_col_idx += y_clinician_col

            # y_clinician_col_idx += self.classes[y_clinician]
            # y_clinician_row_idx += [row_index]

            # y
            y_col = [self.classes[cls] for cls in y.split("|")]
            y_row_idx += [row_index] * len(y_col)
            y_col_idx += y_col

            # sparse entries
            for val_str in x:
                try:
                    col_index, val = val_str.split(":")
                    col_index = int(col_index)
                    val = self.parse_x(val)
                except ValueError as e:
                    print val_str
                    raise e

                x_row_idx.append(row_index)
                x_col_idx.append(col_index)
                x_val.append(val)

            row_index += 1
            if row_index == self.batch_size:
                break

        if row_index == 0:
            raise StopIteration

        # import ipdb; ipdb.set_trace()
        self.X[x_row_idx, x_col_idx] = x_val
        self.Y[y_row_idx, y_col_idx] = 1
        self.Y_clinician[y_clinician_row_idx, y_clinician_col_idx] = 1

        self.patient_count += len(set(patient_ids))

        return self.X, self.Y, self.Y_clinician, patient_ids, row_index
        # assert row_index == self.batch_size

    def parse_x(self, val):
        if self.binarize:
            if self.x_data_type is int:
                return val != "0"
            else:
                return val != "0.0"
        else:
            return self.x_data_type(val)
