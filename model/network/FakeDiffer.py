import mindspore.nn as nn
from .ReconstructionXception import ReconstructionXception
from .Differ import Differ


class FakeDiffer(nn.Cell):
    """ End-to-End Reconstruction-Classification Learning for Face Forgery Detection """

    def __init__(self, num_classes):
        super(FakeDiffer, self).__init__()
        self.num_classes = num_classes
        self.model_reconstruction = ReconstructionXception()
        self.model_differ = Differ(self.num_classes, self.model_reconstruction.encoder)

    def construct(self, in_I, Y, stage):  # torch.Size([2, 3, 299, 299])
        x_recons_real, y_recons_fake = None, None
        if stage == "train":
            x_recons_real, y_recons_fake, out_deep = self.model_reconstruction(in_I, Y, "reconstruction")
            x_recons, y_recons, out_deep = self.model_reconstruction(in_I, Y, "differ")
            Y_pre = self.model_differ(in_I, x_recons, y_recons, out_deep)
            return x_recons_real, y_recons_fake, Y_pre
        else:
            x_recons, y_recons, out_deep = self.model_reconstruction(in_I, Y, "differ")
            Y_pre = self.model_differ(in_I, x_recons, y_recons, out_deep)
            return x_recons_real, y_recons_fake, Y_pre
