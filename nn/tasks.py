from .metrics import multilabel_performance_labels

# Implement modeling task.
from .neural_net.feed_forward import MultiLabelFeedForward


class Task(object):

    def __init__(self):
        pass

    def save(self):
        pass

    def load(self):
        pass

    def execute(self):
        pass

    def iterate(self):
        pass


class SGDFeedForwardTask(Task):

    def __init__(self, data, model, inference, epochs=100):
        self.data = data
        self.model = model
        self.inference = inference
        self.num_epochs = epochs
        self.epoch = 0
        self.print_params()

    def iterate(self):
        self.inference.iterate()
        self.epoch += 1

    def execute(self):
        while self.epoch < self.num_epochs:
            self.iterate()

    def print_params(self):

        print("")
        print("-"*100)
        print("L2_reg", self.model.L2_reg)
        print("param_scale", self.model.param_scale)
        print("learning_rate", self.inference.learning_rate)
        print("momentum", self.inference.momentum)
        print("batch_size", self.inference.batch_size)
        print("num_epochs", self.num_epochs)
        print("hidden", self.model.hidden_layer_sizes)
        print("-"*100)
        print("")

        print("")
        print("-"*100)
        print("num_features", self.data.num_features)
        print("num_labels", self.data.num_labels)
        print("X batch", self.data.X.shape, self.data.X.dtype)
        print("Y batch", self.data.Y.shape, self.data.Y.dtype)
        print("train rows", self.data.info["num_rows_train"])
        print("test rows", self.data.info["num_rows_test"])
        print("ho rows", self.data.info["num_rows_ho"])
        print("-"*100)
        print("")

    def print_perf(self, type="all", test_data=None):
        self.model.data.reset()
        train_X, train_Y = self.model.data.next()

        if test_data is None:
            test_X, test_Y = self.model.data.X_test, self.model.data.Y_test
        else:
            test_data.reset()
            test_X, test_Y = test_data.next()

        train_perf = self.model.error(self.model.W, train_X, train_Y)
        test_perf = self.model.error(self.model.W, test_X, test_Y)

        print "EPOCH", self.epoch, "-"*100

        print "LOSS", self.model.loss(self.model.W, train_X, train_Y)
        # print "Y", self.model.data.Y_train
        # print "C", self.model._predictions(self.model.W, self.model.data.X_train)

        if type == "all":
            labels = multilabel_performance_labels()
            def print_set(ss):
                for i, label in enumerate(labels):
                    print label, ss[i]
            print "TRAIN"
            print_set(train_perf)
            print "TEST"
            print_set(test_perf)
            print
        elif type == "errors":
            print "TRAIN", "TYPE  I ERROR", train_perf[1]
            print "TEST ", "TYPE  I ERROR", test_perf[1]
            print "TRAIN", "TYPE II ERROR", train_perf[3]
            print "TEST ", "TYPE II ERROR", test_perf[3]
