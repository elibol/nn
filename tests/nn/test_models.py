import unittest
import itertools

import sys, logging

import collections
import shutil

import numpy as np
import ipdb

from sklearn import datasets

# from datapipe.utils.sparse import SparseWriter, SparseBatchIterator, SparseBatch
# from datapipe.utils.iter import BatchIterator


from fixtures import multilabel_data

from nn.tasks import SGDFeedForwardTask

from nn.data import Data
from nn.inference.sgd import SGD
from nn.neural_net.feed_forward import FeedForward, MultiLabelFeedForward, BernoulliMultiLabelFeedForward
from nn.neural_net import activations, preprocessing

from nn import utils


class NNTestCase(unittest.TestCase):

    def get_random_XY(self, hidden_layer_sizes=None):
        num_rows = 500
        num_feats = 100
        num_labels = 10

        num_nonzero_X = num_rows * num_feats * 0.1

        # dist = (np.random.binomial, {"n": 10, "p": 0.5})
        # dist = (np.random.normal, {"loc": 0, "scale": 1})
        # dist = (np.random.gamma, {"shape": 6})
        dist = (np.random.uniform, {"low": -1, "high": 1})
        X, Y = multilabel_data(num_rows=num_rows,
                               num_feats=num_feats,
                               num_labels=num_labels,

                               num_nonzero_X=num_nonzero_X,

                               hidden_layer_sizes=hidden_layer_sizes,
                               dist=dist,)
        return X, Y

    def get_params(self):
        hidden_layer_sizes = [100]
        X, Y = self.get_random_XY(hidden_layer_sizes=hidden_layer_sizes)
        return X, Y, hidden_layer_sizes

    def get_iris(self, num_classes=10):
        digits = datasets.load_digits(n_class=num_classes)
        n_samples = len(digits.images)
        X = digits.images.reshape((n_samples, -1))
        num_features = X.shape[1]
        Y = np.zeros(n_samples*num_classes, dtype=np.bool).reshape((n_samples, num_classes))
        for i, val in enumerate(digits.target):
            Y[i][val] = True
        return X, Y

    @unittest.skip("ignore")
    def test_iris_multilabel(self):
        X, Y = self.get_iris(num_classes=10)
        X = utils.standardize(X)
        Y = Y*2 - 1

        flat_index = np.argmax(X)
        row = flat_index/X.shape[1]
        col = flat_index - row*X.shape[1]
        print row, col, X[row][col]

        data = Data(X, Y)
        data.split()

        # model = BernoulliMultiLabelFeedForward(data,
        model = MultiLabelFeedForward(data,
                                      param_scale=0.01,
                                      hidden_activation=activations.relu,
                                      output_activation=activations.identity,
                                      # hidden_layer_sizes=[100]
                                      )

        inference = SGD(model, data,
                        batch_size=1000,
                        learning_rate=1e-3,
                        momentum=0.9,)

        task = SGDFeedForwardTask(data, model, inference)

        task.print_perf("errors")
        while True:
            task.iterate()
            task.print_perf("errors")

    # @unittest.skip("ignore")
    def test_iris_ff(self):
        X, Y = self.get_iris(num_classes=10)

        # X = preprocessing.standardize(X)
        # Y = Y*2 - 1

        data = Data(X, Y, ho_frac=0)
        data.split()

        # U, S, V = preprocessing.SVD(data.X_train)

        # decorrelate
        # data.X_train = np.dot(data.X_train, U)
        # data.X_test = np.dot(data.X_test, U)

        # decorrelate (reduce)
        # data.X_train = np.dot(data.X_train, U[:, :100])
        # data.X_test = np.dot(data.X_test, U[:, :100])

        # whiten
        # data.X_train = data.X_train / np.sqrt(S + 1e-5)
        # data.X_test = data.X_test / np.sqrt(S + 1e-5)

        model = FeedForward(data,
                            param_scale=0.01,
                            hidden_activation=activations.relu,
                            output_activation=activations.softmax,
                            hidden_layer_sizes=[64],
                            # dropout_rate=0.05,
                            )

        inference = SGD(model, data,
                        batch_size=1000,
                        learning_rate=1e-3,
                        momentum=0.9,)

        task = SGDFeedForwardTask(data, model, inference)

        task.inference.quick_loss_check()
        # task.inference.quick_grad_check()

        task.print_perf("errors")
        while True:
            task.iterate()
            task.print_perf("errors")

    @unittest.skip("ignore")
    def test_label_ff(self):
        X, Y, hidden_layer_sizes = self.get_params()
        data = Data(X, Y)
        data.split()

        model = FeedForward(data,
                            param_scale=0.01,
                            hidden_activation=activations.relu,
                            output_activation=activations.softmax,
                            hidden_layer_sizes=hidden_layer_sizes)

        inference = SGD(model, data,
                        batch_size=1000,
                        learning_rate=1e-3,
                        momentum=0.9,)

        task = SGDFeedForwardTask(data, model, inference)

        task.print_perf("errors")
        while True:
            task.iterate()
            task.print_perf("errors")

        # EPOCH 229 ----------------------------------------------------------------------------------------------------
        # LOSS 6728.70956894
        # TRAIN TYPE  I ERROR 0.0418962585034
        # TEST  TYPE  I ERROR 0.0990952380952
        # TRAIN TYPE II ERROR 0.0418962585034
        # TEST  TYPE II ERROR 0.0990952380952

    @unittest.skip("ignore")
    def test_label_multilabel(self):
        X, Y, hidden_layer_sizes = self.get_params()

        # X = utils.standardize(X)

        data = Data(X, Y, sparse=False)
        data.split()

        # model = MultiLabelFeedForward(data,
        model = BernoulliMultiLabelFeedForward(data,
                                      param_scale=0.01,
                                      hidden_activation=activations.relu,
                                      output_activation=activations.identity,
                                      hidden_layer_sizes=hidden_layer_sizes,
                                      dropout_rate=None)

        inference = SGD(model, data,
                        batch_size=1000,
                        learning_rate=1e-3,
                        momentum=0.9,)

        task = SGDFeedForwardTask(data, model, inference)

        task.print_perf("errors")
        while True:
            task.iterate()
            task.print_perf("errors")

        # EPOCH 118 ----------------------------------------------------------------------------------------------------
        # LOSS 61.1996235829
        # TRAIN TYPE  I ERROR 0.00553741496599
        # TEST  TYPE  I ERROR 0.0559285714286
        # TRAIN TYPE II ERROR 0.00553741496599
        # TEST  TYPE II ERROR 0.0559285714286
