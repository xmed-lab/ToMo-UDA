# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from bisect import bisect_right

def make_optimizer(cfg, model, name):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        if name == 'backbone':
            lr = cfg.SOLVER.BACKBONE.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            if "bias" in key:
                lr = cfg.SOLVER.BACKBONE.BASE_LR * cfg.SOLVER.BACKBONE.BIAS_LR_FACTOR
                weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        elif name == 'fcos':
            lr = cfg.SOLVER.FCOS.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            if "bias" in key:
                lr = cfg.SOLVER.FCOS.BASE_LR * cfg.SOLVER.FCOS.BIAS_LR_FACTOR
                weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        elif name == 'roi_head':
            lr = cfg.SOLVER.FCOS.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            if "bias" in key:
                lr = cfg.SOLVER.FCOS.BASE_LR * cfg.SOLVER.FCOS.BIAS_LR_FACTOR
                weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS

        elif name == 'middle_head':
            lr = cfg.SOLVER.MIDDLE_HEAD.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            if "bias" in key:
                lr = cfg.SOLVER.MIDDLE_HEAD.BASE_LR * cfg.SOLVER.MIDDLE_HEAD.BIAS_LR_FACTOR
                weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        elif name == 'discriminator':
            lr = cfg.SOLVER.DIS.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            if "bias" in key:
                lr = cfg.SOLVER.DIS.BASE_LR * cfg.SOLVER.DIS.BIAS_LR_FACTOR
                weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        else:
            raise AssertionError('here')
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    # if cfg.SOLVER.OPTIMIZER == 'SGD':
    optimizer = torch.optim.SGD(params, lr, momentum=cfg.SOLVER.MOMENTUM)
    # elif cfg.SOLVER.OPTIMIZER == 'Adam':
    #     optimizer = torch.optim.Adam(params, lr)



    return optimizer


def make_lr_scheduler(cfg, optimizer, name):
    if name == 'backbone':
        return WarmupMultiStepLR(
            optimizer,
            cfg.SOLVER.BACKBONE.STEPS,
            cfg.SOLVER.BACKBONE.GAMMA,
            warmup_factor=cfg.SOLVER.BACKBONE.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.BACKBONE.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.BACKBONE.WARMUP_METHOD,
        )
    elif name == 'fcos':
        return WarmupMultiStepLR(
            optimizer,
            cfg.SOLVER.FCOS.STEPS,
            cfg.SOLVER.FCOS.GAMMA,
            warmup_factor=cfg.SOLVER.FCOS.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.FCOS.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.FCOS.WARMUP_METHOD,
        )
    elif name == 'middle_head':
        return WarmupMultiStepLR(
            optimizer,
            cfg.SOLVER.MIDDLE_HEAD.STEPS,
            cfg.SOLVER.MIDDLE_HEAD.GAMMA,
            warmup_factor=cfg.SOLVER.MIDDLE_HEAD.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.MIDDLE_HEAD.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.MIDDLE_HEAD.WARMUP_METHOD,
        )
    elif name == 'discriminator':
        return WarmupMultiStepLR(
            optimizer,
            cfg.SOLVER.DIS.STEPS,
            cfg.SOLVER.DIS.GAMMA,
            warmup_factor=cfg.SOLVER.DIS.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.DIS.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.DIS.WARMUP_METHOD,
        )
    elif name == 'roi_head':
        return WarmupMultiStepLR(
            optimizer,
            cfg.SOLVER.FCOS.STEPS,
            cfg.SOLVER.FCOS.GAMMA,
            warmup_factor=cfg.SOLVER.FCOS.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.FCOS.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.FCOS.WARMUP_METHOD,
        )
    else:
        raise AssertionError('here')

# FIXME ideally this would be achieved with a CombinedLRScheduler,
# separating MultiStepLR with WarmupLR
# but the current LRScheduler design doesn't allow it
class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        warmup_method="linear",
        last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]
