from mindspore import nn


def get_loss(stage, name="cross_entropy", device="cuda:0"):
    print(f"{stage} Using loss: '{LOSSES[name]}'")
    return LOSSES[name].to(device)


LOSSES = {"binary_ce": nn.BCEWithLogitsLoss(), "cross_entropy": nn.CrossEntropyLoss(), "mse": nn.MSELoss()}
