import os
import sys
import time
import math
import yaml
import mindspore as ms
import random

from tqdm import tqdm
from pprint import pprint
# from tensorboardX import SummaryWriter
from mindspore.train import SummaryCollector

# ============= self-definited =============
from dataset import load_dataset
from loss import get_loss
from model import load_model
from optimizer import get_optimizer
from scheduler import get_scheduler
from trainer import AbstractTrainer, LEGAL_METRIC
from trainer.utils import exp_recons_loss, MLLoss, reduce_tensor, center_print
from trainer.utils import MODELS_PATH, AccMeter, AUCMeter, AverageMeter, Logger, Timer


def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr


class ModelTrainer(AbstractTrainer):
    def __init__(self, config, stage="Train"):
        super(ModelTrainer, self).__init__(config, stage)

    def _mprint(self, content=""):
        print(content)

    def _initiated_settings(self, model_cfg, data_cfg, config_cfg):
        pass

    def _train_settings(self, model_cfg, data_cfg, config_cfg):
        # debug mode: no log dir, no train_val operation.
        self.debug = config_cfg["debug"]
        self._mprint(f"Using debug mode: {self.debug}.")
        self._mprint("*" * 20)

        self.eval_metric = config_cfg["metric"]
        if self.eval_metric not in LEGAL_METRIC:
            raise ValueError(f"Evaluation metric must be in {LEGAL_METRIC}, but found "
                             f"{self.eval_metric}.")
        else:
            self._mprint(f"Using the metric: {self.eval_metric}.")

        if self.eval_metric == LEGAL_METRIC[-1]:
            self.best_metric = 1.0e8

        # load training and val dataset
        file_dataset = data_cfg["file"]
        data_name = data_cfg["name"]
        self.train_batch_size = data_cfg["train_batch_size"]
        self.val_batch_size = data_cfg["val_batch_size"]
        with open(file_dataset, "r") as f:
            options = yaml.load(f, Loader=yaml.FullLoader)

        train_options = options[data_cfg["train_branch"]]
        self.train_set = load_dataset(data_name)(train_options)
        self.train_loader = data.DataLoader(self.train_set, shuffle=True, num_workers=data_cfg.get("num_workers", 4),
                                            batch_size=self.train_batch_size)

        val_options = options[data_cfg["val_branch"]]
        self.val_set = load_dataset(data_name)(val_options)
        self.val_loader = data.DataLoader(self.val_set, shuffle=False, num_workers=data_cfg.get("num_workers", 4),
                                          batch_size=self.val_batch_size)

        self.load_pretrain = config_cfg.get("load_pretrain", False)

        if not self.debug:
            if not self.load_pretrain:
                time_format = "%Y-%m-%d-%H:%M:%S"
                run_time = time.strftime(time_format, time.localtime(time.time()))
                self.run_id = config_cfg.get("id", run_time)
                self.dir = os.path.join(f"runs_lr_{self.config['config']['optimizer']['lr']}_" + run_time, self.model_name, self.run_id)

                if os.path.exists(self.dir):
                    raise ValueError("Error: given id '%s' already exists." % self.run_id)
                os.makedirs(self.dir, exist_ok=True)
                print(f"Writing config file to file directory: {self.dir}.")
                yaml.dump({"config": self.config,
                           "train_data": train_options,
                           "val_data": val_options},
                          open(os.path.join(self.dir, 'train_config.yml'), 'w'))
                # copy the script for the training model
                model_file = MODELS_PATH[self.model_name]

                os.system("cp " + model_file + " " + self.dir)
            else:
                time_format = "%Y-%m-%d-%H:%M:%S"
                run_time = time.strftime(time_format, time.localtime(time.time()))

                self.pretrain_dir = config_cfg.get("pretrain_dir", None)
                self.pretrain_pth = config_cfg.get("pretrain_pth", None)
                print(f"The pretrain_pth file is in directory: {self.pretrain_pth}.")

                self.dir = os.path.join(self.pretrain_dir, "runs_pretrain_" + run_time)
                self.pretrain_model = os.path.join(self.pretrain_dir, self.pretrain_pth)

                if os.path.exists(self.dir):
                    raise ValueError("Error: given id '%s' already exists." % self.run_id)
                os.makedirs(self.dir, exist_ok=True)
                print(f"Writing config file to file directory: {self.dir}.")
                yaml.dump({"config": self.config,
                           "train_data": train_options,
                           "val_data": val_options},
                          open(os.path.join(self.dir, 'train_config.yml'), 'w'))

            print(f"Logging directory: {self.dir}.")

            # redirect the std out stream
            sys.stdout = Logger(os.path.join(self.dir, 'records.txt'))
            center_print('Train configurations begins.')
            pprint(self.config)
            pprint(train_options)
            pprint(val_options)

            center_print('Train configurations ends.')

        # load model
        # self.device = "cuda"  # #######################################################################
        # self.model_reconstruction = load_model(self.model_reconstruction_name)(**model_cfg)
        # self.model_reconstruction = load_model(self.model_reconstruction_name)(self.recons_initials)
        self.model = load_model(self.model_name)(self.num_classes)
        if self.load_pretrain:
            self.model.load_state_dict(torch.load(self.pretrain_model)["model"])
            self.model = self.model.to(self.device)
        else:
            self.model = self.model.to(self.device)

        torch.distributed.init_process_group(backend='nccl', init_method='tcp://localhost:23456', world_size=1, rank=0)

        # load optimizer
        optim_cfg = config_cfg.get("optimizer", None)
        optim_name = optim_cfg.pop("name")
        self.optimizer = get_optimizer(optim_name)(self.model.parameters(), **optim_cfg)
        
        # load scheduler
        self.scheduler = get_scheduler(self.optimizer, config_cfg.get("scheduler", None))
        # load loss
        self.recons_loss_criterion = get_loss("Reconstruction", config_cfg.get("recons_loss", None), device=self.device)
        self.classify_loss_criterion = get_loss("Classify", config_cfg.get("classify_loss", None), device=self.device)

        self.num_epoch = config_cfg.get('num_epoch', 50)

        # the number of steps to write down a log
        self.log_steps = train_options["log_steps"]  # #####################################
        # the number of steps to validate on val dataset once
        self.val_steps = train_options["val_steps"]

        self.num_steps = config_cfg["num_steps"]  # #####################################

        # balance coefficients
        self.lambda_1 = config_cfg["lambda_1"]  # ######## can be used in fake and real weight reconstruction ##########
        self.lambda_2 = config_cfg["lambda_2"]  # #####################################
        self.warmup_step = config_cfg.get('warmup_step', 0)
        # print("166 self.recons_warmup_step :", self.recons_warmup_step)

        self.acc_meter = AccMeter()
        self.loss_meter = AverageMeter()
        # self.classify_loss_meter = AverageMeter()  # ##############################################
        self.recons_loss_meter = AverageMeter()

        # if self.resume and self.local_rank == 0:  # ############ can be deleted ################
        # if self.resume:
        #     self._load_ckpt(best=config_cfg.get("resume_best", False), train=True)

    def _test_settings(self, model_cfg, data_cfg, config_cfg):
        # Not used.
        raise NotImplementedError("The function is not intended to be used here.")

    def _load_ckpt(self, best=False, train=False):
        # Not used.
        raise NotImplementedError("The function is not intended to be used here.")

    def _save_ckpt(self, epoch, step, best=False):
        save_dir = os.path.join(self.dir, f"best_model_{step}.pth" if best else f"latest_model.pth")
        torch.save({"epoch": epoch, "step": step, "best_step": self.best_step, "best_metric": self.best_metric,
                    "eval_metric": self.eval_metric, "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict()}, save_dir)

    def train(self):
        # try:
        timer = Timer()
        grad_scalar = GradScaler(2 ** 10)
        # if self.local_rank == 0:
        writer = None if self.debug else SummaryWriter(log_dir=self.dir)
        center_print("Training begins......")
        if self.load_pretrain:
            start_epoch = torch.load(self.pretrain_model)["epoch"] + 1
        else:
            start_epoch = self.start_epoch

        self.model.train()

        step = 0
        loss_min = 1e8
        for epoch_idx in range(start_epoch, self.num_epoch + 1):
            # reset meter
            self.acc_meter.reset()
            # self.classify_loss_meter.reset()
            self.loss_meter.reset()
            self.recons_loss_meter.reset()

            train_generator = enumerate(self.train_loader, 1)
            # wrap train generator with tqdm for process 0
            train_generator = tqdm(train_generator, position=0, leave=True)

            for batch_idx, train_data in train_generator:
                step += 1
                I, Y = train_data
                # I = self.train_loader.dataset.load_item(I)  # function in abstract_dataset.py
                in_I, Y = self.to_device((I, Y))

                # warm-up lr
                if self.warmup_step != 0 and step <= self.warmup_step:
                    lr = self.config['config']['optimizer']['lr'] * float(step) / self.warmup_step
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr
                elif self.warmup_step == 0 or step > (self.warmup_step + self.start_decay_after):
                    lr = self.config['config']['optimizer']['lr'] * (0.9 ** ((step - self.warmup_step-self.start_decay_after) // self.decay_step_size + 1))
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr
                    # self.scheduler.step()
                else:
                    lr = self.config['config']['optimizer']['lr']
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr

                self.optimizer.zero_grad()
                with autocast():
                    x_recons_real, y_recons_fake, Y_pre = self.model(in_I, Y, "train")
                    recons_loss = exp_recons_loss(x_recons_real, y_recons_fake, (in_I, Y), self.lambda_1, self.lambda_2)  # 0.6763

                    if self.num_classes == 1:
                        Y_pre = Y_pre.squeeze()
                        classify_loss = self.classify_loss_criterion(Y_pre, Y.float())
                        Y_pre = torch.sigmoid(Y_pre)
                    else:
                        classify_loss = self.classify_loss_criterion(Y_pre, Y)

                    classify_loss = (classify_loss - 0.04).abs() + 0.04

                    loss = classify_loss + recons_loss

                grad_scalar.scale(loss).backward()
                grad_scalar.step(self.optimizer)
                grad_scalar.update()

                self.acc_meter.update(Y_pre, Y, self.num_classes == 1)
                self.loss_meter.update(reduce_tensor(loss).item())
                self.recons_loss_meter.update(reduce_tensor(recons_loss).item())
                iter_acc = reduce_tensor(self.acc_meter.mean_acc()).item()

                if self.loss_meter.avg < loss_min:
                    loss_min = self.loss_meter.avg

                if step % self.log_steps == 0 and writer is not None:
                    writer.add_scalar("train/Acc", iter_acc, step)
                    writer.add_scalar("train/Loss", self.loss_meter.avg, step)
                    writer.add_scalar("train/Recons_Loss", self.recons_loss_meter.avg, step)
                    writer.add_scalar("train/LR :", self.optimizer.state_dict()['param_groups'][0]['lr'], step)

                # log training step
                train_generator.set_description(
                    "Train Epoch %d (%d/%d), Step %d, Loss %.4f, Loss_min %.4f,  Recons Loss %.4f, ACC %.4f, LR %.6f"
                    % (epoch_idx, batch_idx, len(self.train_loader), step, self.loss_meter.avg, loss_min,
                       self.recons_loss_meter.avg, iter_acc, self.optimizer.state_dict()['param_groups'][0]['lr']))

                # validating process
                if step % self.val_steps == 0 and not self.debug:
                    print()
                    self.validate(epoch_idx, step, timer, writer)

                # # when num_steps has been set and the training process will
                # # be stopped earlier than the specified num_epochs, then stop.
                # if self.num_steps_recons is not None and recons_step == self.num_steps_recons:
                #     if writer is not None:
                #         writer.close()
                #     # if self.local_rank == 0:
                #     print()
                #     center_print("Training reconstruction process ends.")
                #     return
            train_generator.close()
            print()

        if writer is not None:
            writer.close()
        center_print("Training process ends.")

    def validate(self, epoch, step, timer, writer):
        # v_idx = random.randint(1, len(self.val_loader) + 1)
        # categories = self.val_loader.dataset.categories
        self.model.eval()
        with torch.no_grad():
            acc = AccMeter()
            auc = AUCMeter()
            loss_meter = AverageMeter()
            cur_acc = 0.0  # Higher is better
            cur_auc = 0.0  # Higher is better
            cur_loss = 1e8  # Lower is better
            val_generator = tqdm(enumerate(self.val_loader, 1), position=0, leave=True)
            for val_idx, val_data in val_generator:
                I, Y = val_data
                # I = self.val_loader.dataset.load_item(I)
                in_I, Y = self.to_device((I, Y))
                # Y_pre = self.model(in_I)
                x_recons_real, y_recons_fake, Y_pre = self.model(in_I, Y, "val")

                # for BCE Setting:
                if self.num_classes == 1:
                    Y_pre = Y_pre.squeeze()
                    classify_loss = self.classify_loss_criterion(Y_pre, Y.float())
                    Y_pre = torch.sigmoid(Y_pre)
                else:
                    classify_loss = self.classify_loss_criterion(Y_pre, Y)

                acc.update(Y_pre, Y, self.num_classes == 1)
                auc.update(Y_pre, Y, self.num_classes == 1)
                loss_meter.update(classify_loss.item())

                cur_acc = acc.mean_acc()
                cur_loss = loss_meter.avg

                val_generator.set_description("Eval Epoch %d (%d/%d), Step %d, Loss %.4f, ACC %.4f" %
                                              (epoch, val_idx, len(self.val_loader), step, cur_loss, cur_acc))

                # if val_idx == v_idx or val_idx == 1:
                #     sample_recons = list()
                #     # for _ in self.model.module.loss_inputs['recons']:
                #     for _ in self.model.loss_inputs['recons']:
                #         sample_recons.append(_[:4].to("cpu"))
                #     # show images
                #     images = I[:4]
                #     images = torch.cat([images, *sample_recons], dim=0)
                #     pred = Y_pre[:4]
                #     gt = Y[:4]
                #     figure = self.plot_figure(images, pred, gt, 4, categories, show=False)

            cur_auc = auc.mean_auc()
            print("Eval Epoch %d, Loss %.4f, ACC %.4f, AUC %.4f" % (epoch, cur_loss, cur_acc, cur_auc))
            if writer is not None:
                writer.add_scalar("val/Loss", cur_loss, step)
                writer.add_scalar("val/Acc", cur_acc, step)
                writer.add_scalar("val/AUC", cur_auc, step)
                # writer.add_figure("val/Figures", figure, step)
            # record the best acc and the corresponding step
            if self.eval_metric == 'Acc' and cur_acc >= self.best_metric:
                self.best_metric = cur_acc
                self.best_step = step
                self._save_ckpt(epoch, step, best=True)
            elif self.eval_metric == 'AUC' and cur_auc >= self.best_metric:
                self.best_metric = cur_auc
                self.best_step = step
                self._save_ckpt(epoch, step, best=True)
            elif self.eval_metric == 'LogLoss' and cur_loss <= self.best_metric:
                self.best_metric = cur_loss
                self.best_step = step
                self._save_ckpt(epoch, step, best=True)
            print("Best Step %d, Best %s %.4f, Running Time: %s, Estimated Time: %s" % (
                self.best_step, self.eval_metric, self.best_metric, timer.measure(),
                timer.measure(step / self.num_steps)))
            self._save_ckpt(epoch, step, best=False)

    def test(self):
        # Not used.
        raise NotImplementedError("The function is not intended to be used here.")
