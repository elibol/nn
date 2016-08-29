from autograd.util import quick_grad_check
import autograd.numpy as np

from nn import utils

from scipy.optimize import line_search


class SGD(object):

    def __init__(self,
                 model,
                 data,
                 learning_rate=1e-3,
                 momentum=0.9,
                 batch_size=1000,):

        self.model = model
        self.data = data
        self.X_train = data.X_train
        self.Y_train = data.Y_train

        # play around with increasing this
        self.learning_rate = learning_rate
        # momentum = 1-500.0 / 4869.0
        self.momentum = momentum
        # variance of gradient vs. computation time
        self.batch_size = batch_size
        if self.batch_size > self.X_train.shape[0]:
            self.batch_size = self.X_train.shape[0]
            print("set batch_sizes to %s" % self.batch_size)

        self.batch_idxs = self.make_batches(self.X_train.shape[0], self.batch_size)
        self.X_batch = np.zeros(shape=(self.batch_size, self.X_train.shape[1]), dtype=self.X_train.dtype)
        self.Y_batch = np.zeros(shape=(self.batch_size, self.Y_train.shape[1]), dtype=self.Y_train.dtype)

        self.cur_dir = np.zeros(self.model.N)

        self.epochs = 0

    def quick_grad_check(self):
        idxs = self.batch_idxs[0]

        row, col = self.X_train[idxs].nonzero()
        self.X_batch[:] = 0
        self.X_batch[row, col] = self.get_X_batch(idxs, row, col)

        row, col = self.Y_train[idxs].nonzero()
        self.Y_batch[:] = 0
        self.Y_batch[row, col] = self.get_Y_batch(idxs, row, col)
        quick_grad_check(self.model.loss_grad, self.model.W, (self.X_batch, self.Y_batch))

    def quick_loss_check(self):
        idxs = self.batch_idxs[0]

        row, col = self.X_train[idxs].nonzero()
        self.X_batch[:] = 0
        self.X_batch[row, col] = self.get_X_batch(idxs, row, col)

        row, col = self.Y_train[idxs].nonzero()
        self.Y_batch[:] = 0
        self.Y_batch[row, col] = self.get_Y_batch(idxs, row, col)
        self.model.loss(self.model.W, self.X_batch, self.Y_batch)

    def iterate(self):
        # print("batches %s" % len(batch_idxs))
        for idxs in self.batch_idxs:

            # import ipdb; ipdb.set_trace()

            row, col = self.X_train[idxs].nonzero()
            self.X_batch[:] = 0
            self.X_batch[row, col] = self.get_X_batch(idxs, row, col)

            row, col = self.Y_train[idxs].nonzero()
            self.Y_batch[:] = 0
            self.Y_batch[row, col] = self.get_Y_batch(idxs, row, col)

            # import ipdb; ipdb.set_trace()

            # line search imp
            
            grad_W = self.model.loss_grad(self.model.W, self.X_batch, self.Y_batch)
            
            self.cur_dir = self.momentum * self.cur_dir + (1.0 - self.momentum) * grad_W
            
            alpha, fc, gc, new_fval, old_fval, new_slope = line_search(
                self.model.loss,
                self.model.loss_grad, 
                self.model.W, 
                - self.cur_dir, 
                # grad_W,
                args=(self.X_batch, self.Y_batch))
            if alpha is None:
                print "line search did not converge"
                alpha = self.learning_rate
            self.model.W -= alpha * self.cur_dir
            
            """
            grad_W = self.model.loss_grad(self.model.W, self.X_batch, self.Y_batch)

            # zero out dropout layers
            # grad_W[self.model.dropout_mask] = 0
            
            
            self.cur_dir = self.momentum * self.cur_dir + (1.0 - self.momentum) * grad_W
            self.model.W -= self.learning_rate * self.cur_dir

            # multiple matrices here
            # self.model.W[self.model.dropout_mask] = np.random.binomial(1, 1 - self.dropout_rate, np.sum(self.model.dropout_mask == True))
            """
            
        self.epochs += 1

    def make_batches(self, N_data, batch_size):
        return [slice(i, min(i+batch_size, N_data))
                for i in range(0, N_data, batch_size)]

    def get_X_batch(self, idxs, row, col):
        return self.X_train[idxs][row, col]

    def get_Y_batch(self, idxs, row, col):
        return self.Y_train[idxs][row, col]


class BatchNormalizedSGD(SGD):

    def get_X_batch(self, idxs, row, col):
        return utils.standardize(self.X_train[idxs][row, col])

    def get_Y_batch(self, idxs, row, col):
        return self.Y_train[idxs][row, col]


class BatchSGD(object):

    def __init__(self,
                 model,
                 data,
                 learning_rate=1e-3,
                 momentum=0.9,):

        self.model = model
        self.data = data

        # play around with increasing this
        self.learning_rate = learning_rate
        # momentum = 1-500.0 / 4869.0
        self.momentum = momentum

        self.cur_dir = np.zeros(self.model.N)

        self.epochs = 0

    def quick_grad_check(self):
        self.data.reset()
        X, Y, patient_id, num_rows = self.data.next()
        self.data.reset()
        quick_grad_check(self.model.loss_grad, self.model.W, (X, Y))

    def iterate(self):
        for X, Y, patient_id, num_rows in self.data:
            # import ipdb; ipdb.set_trace()
            grad_W = self.model.loss_grad(self.model.W, X, Y)
            self.cur_dir = self.momentum * self.cur_dir + (1.0 - self.momentum) * grad_W
            self.model.W -= self.learning_rate * self.cur_dir

        self.epochs += 1
