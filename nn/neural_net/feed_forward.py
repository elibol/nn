import itertools
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad

from nn.neural_net.activations import *
from nn.metrics import singlelabel_error, multilabel_performance


# TODO: Test difference between += and regular +

class FeedForward(object):

    def __init__(self,
                 data,
                 hidden_layer_sizes=None,
                 L2_reg=1.0,
                 param_scale=0.1,
                 hidden_activation=sigmoid,
                 output_activation=softmax,
                 dropout_rate=None):

        self.dropout_rate = dropout_rate

        self.data = data

        self.hidden_layer_sizes = []
        if hidden_layer_sizes is not None:
            self.hidden_layer_sizes = hidden_layer_sizes

        self.layer_sizes = [self.data.num_features] + self.hidden_layer_sizes + [self.data.num_labels]

        # add parameter dropout index
        # e.g. 1
        # self.layer_sizes
        # dropout_index = [1]
        # self.layer_sizes = self.layer_sizes[dropout_index]

        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

        self.param_scale = param_scale
        self.L2_reg = L2_reg

        self.shapes = zip(self.layer_sizes[:-1], self.layer_sizes[1:])
        self.N = sum((m+1)*n for m, n in self.shapes)
        rs = npr.RandomState()
        self.W = rs.randn(self.N) * self.param_scale

        # TODO: set W binomial for dropouts here

        self.loss_grad = grad(self.loss, 0)

    def _unpack_layers(self, W_vect):
        for m, n in self.shapes:
            yield W_vect[:m*n].reshape((m,n)), W_vect[m*n:m*n+n]
            W_vect = W_vect[(m+1)*n:]

    def _predictions(self, W_vect, inputs):
        for W, b in self._unpack_layers(W_vect):
            outputs = np.dot(inputs, W) + b
            inputs = self.hidden_activation(outputs)

            # dropout
            if self.dropout_rate is not None:
                inputs = inputs * np.random.binomial(1, 1 - self.dropout_rate, inputs.shape)

        return self.output_activation(outputs)

    def get_parameters(self):
        return self.W

    def set_parameters(self, W):
        self.W = W

    def loss(self, W_vect, X, T):
        log_prior = -self.L2_reg * np.dot(W_vect, W_vect)
        log_lik = np.sum(self._predictions(W_vect, X) * T)
        # import ipdb; ipdb.set_trace()
        return - log_prior - log_lik

    def error(self, W_vect, X, T):
        y_pred = self._predictions(W_vect, X)
        return multilabel_performance(T, y_pred)
        # return singlelabel_error(T, y_pred)


class MultiLabelFeedForward(FeedForward):

    def __init__(self, *args, **kwargs):
        kwargs['output_activation'] = identity
        super(MultiLabelFeedForward, self).__init__(*args, **kwargs)

    def loss(self, W_vect, X, T):
        c = self._predictions(W_vect, X)
        m = T.shape[0]
        total_error = 0.0
        for i in range(m):

            Y = np.where(T[i] > 0)[0]
            if Y.shape[0] == 0:
                continue

            neg_Y = np.where(T[i] < 0)[0]
            if neg_Y.shape[0] == 0:
                continue

            r = 0.0
            for k, l in itertools.product(Y, neg_Y):
                pos_val = c[i][k]
                neg_val = c[i][l]
                val = -(pos_val - neg_val)

                exp_val = np.exp(val)
                r = r + exp_val
            r = r * 1.0/(Y.shape[0]*neg_Y.shape[0])
            total_error = total_error + r
        return total_error

    def error(self, W_vect, X, T):
        Y_pred = self._predictions(W_vect, X)
        return multilabel_performance(T, Y_pred)


class FastMultiLabelFeedForward(MultiLabelFeedForward):

    def loss(self, W_vect, X, T):
        c = self._predictions(W_vect, X)
        T_numeric = T * 2 - 1
        # return logsumexp(-c*T_numeric)
        return np.sum(np.exp(-(c*T_numeric)))

    def loss2(self, X, T, W = None):
        prows, pcols = np.where(T)
        nrows, ncols = np.where(~T)
        # then what?

        # Y = np.repeat(Y, neg_Y, axis=1)


class BernoulliMultiLabelFeedForward(FeedForward):

    def __init__(self, *args, **kwargs):
        kwargs['output_activation'] = identity
        super(BernoulliMultiLabelFeedForward, self).__init__(*args, **kwargs)

    def loss(self, W_vect, X, T):
        c = self._predictions(W_vect, X)
        c = sigmoid(c)
        # log_prior = -self.L2_reg * np.dot(W_vect, W_vect)
        ll = -(np.sum(np.log(c[T == 1])) + np.sum(np.log(1-c[T == -1])))
        # return ll + log_prior
        return ll

    def error(self, W_vect, X, T):
        Y_pred = self._predictions(W_vect, X)
        return multilabel_performance(T, Y_pred)

