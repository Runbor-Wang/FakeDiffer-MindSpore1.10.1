from mindspore.nn.lr_scheduler import _LRScheduler
from mindspore.nn.lr_scheduler import StepLR
from mindspore.nn.lr_scheduler import MultiStepLR
from mindspore.nn.lr_scheduler import ExponentialLR
from mindspore.nn.lr_scheduler import CosineAnnealingLR
from mindspore.nn.lr_scheduler import CosineAnnealingWarmRestarts
from mindspore.nn.lr_scheduler import ReduceLROnPlateau


class ConstantLR(_LRScheduler):
    def __init__(self, optimizer, last_epoch=-1):
        super(ConstantLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr for base_lr in self.base_lrs]


SCHEDULERS = {
    'ConstantLR': ConstantLR,
    "StepLR": StepLR,
    "MultiStepLR": MultiStepLR,
    "CosineAnnealingLR": CosineAnnealingLR,
    "CosineAnnealingWarmRestarts": CosineAnnealingWarmRestarts,
    "ExponentialLR": ExponentialLR,
    "ReduceLROnPlateau": ReduceLROnPlateau
}


def get_scheduler(optimizer, kwargs):
    if kwargs is None:
        print("No lr scheduler is used.")
        return ConstantLR(optimizer)
    name = kwargs["name"]
    kwargs.pop("name")
    print("Using scheduler: '%s' with params: %s" % (name, kwargs))
    return SCHEDULERS[name](optimizer, **kwargs)
