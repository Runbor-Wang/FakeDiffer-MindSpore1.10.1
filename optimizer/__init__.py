from mindspore.nn import SGD
from mindspore.nn import Adam
from mindspore.nn import ASGD
from mindspore.nn import Adadelta
from mindspore.nn import Adagrad

key2opt = {
    'sgd': SGD,
    'adam': Adam,
    'asgd': ASGD,
    'adadelta': Adadelta,
    'adagrad': Adagrad,
}


def get_optimizer(optimizer_name=None):
    if optimizer_name is None:
        print("Using default 'SGD' optimizer")
        return SGD

    else:
        if optimizer_name not in key2opt:
            raise NotImplementedError(f"Optimizer '{optimizer_name}' not implemented")

        print(f"Using optimizer: '{optimizer_name}'")
        return key2opt[optimizer_name]
