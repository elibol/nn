from .neural_net.feed_forward import MultiLabelFeedForward, FastMultiLabelFeedForward
from .tasks import SGDFeedForwardTask
from .inference.sgd import SGD
from .data import Data

sgd_fastmlff_1000 = {

    "num_epochs": 10000000,

    "inference": {
        "class": SGD,
        "learning_rate": 1e-3,
        "momentum": 0.9,
        "batch_size": 1000,
    },

    "model": {
        "class": FastMultiLabelFeedForward,
        "hidden_layer_sizes": [],
        "l2": 1.0,
        "param_scale": 0.1,
    },

    "data": {
        "class": Data,
        "seed": 0,
    },

}
