import os
import mindspore as ms
import random
import numpy as np

LEGAL_METRIC = ['Acc', 'AUC', 'MSE', 'LogLoss']


class AbstractTrainer(object):
    def __init__(self, config, stage="Train"):
        feasible_stage = ["Train", "Test"]
        if stage not in feasible_stage:
            raise ValueError(f"stage should be in {feasible_stage}, but found '{stage}'")

        self.config = config
        model_cfg = config.get("model", None)
        data_cfg = config.get("data", None)
        config_cfg = config.get("config", None)

        self.model_name = model_cfg.pop("model_name")
        self.num_classes = model_cfg.pop("num_classes")

        self.gpu = None
        self.dir = None
        self.debug = None
        # self.device = None
        self.device = "cuda"
        self.resume = None

        self.best_metric = 0.0
        self.best_step = 1
        self.start_step = 1
        self.start_epoch = 1
        self.start_decay_after = config_cfg.get("start_decay_after", None)
        self.decay_step_size = config_cfg['scheduler']['step_size']

        self._initiated_settings(model_cfg, data_cfg, config_cfg)

        if stage == 'Train':
            self._train_settings(model_cfg, data_cfg, config_cfg)
        if stage == 'Test':
            self._test_settings(model_cfg, data_cfg, config_cfg)

    def _initiated_settings(self, model_cfg, data_cfg, config_cfg):
        raise NotImplementedError("Not implemented in abstract class.")

    def _train_settings(self, model_cfg, data_cfg, config_cfg):
        raise NotImplementedError("Not implemented in abstract class.")

    def _test_settings(self, model_cfg, data_cfg, config_cfg):
        raise NotImplementedError("Not implemented in abstract class.")

    def _save_ckpt(self, epoch, step, best=False):
        raise NotImplementedError("Not implemented in abstract class.")

    def _load_ckpt(self, best=False, train=False):
        raise NotImplementedError("Not implemented in abstract class.")

    def to_device(self, items):
        return [obj.to(self.device) for obj in items]

    @staticmethod
    def fixed_randomness():
        random.seed(42)
        np.random.seed(42)

    def train(self):
        raise NotImplementedError("Not implemented in abstract class.")

    def validate(self, epoch, step, timer, writer):
        raise NotImplementedError("Not implemented in abstract class.")

    def test(self):
        raise NotImplementedError("Not implemented in abstract class.")
