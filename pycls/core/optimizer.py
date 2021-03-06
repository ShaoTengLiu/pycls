#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Optimizer."""

import numpy as np
import torch
from pycls.core.config import cfg


def construct_optimizer(model):
    """Constructs the optimizer.

    Note that the momentum update in PyTorch differs from the one in Caffe2.
    In particular,

        Caffe2:
            V := mu * V + lr * g
            p := p - V

        PyTorch:
            V := mu * V + g
            p := p - lr * V

    where V is the velocity, mu is the momentum factor, lr is the learning rate,
    g is the gradient and p are the parameters.

    Since V is defined independently of the learning rate in PyTorch,
    when the learning rate is changed there is no need to perform the
    momentum correction by scaling V (unlike in the Caffe2 case).
    """
    if cfg.OPTIM.PARAMS == 'default':
        if cfg.BN.USE_CUSTOM_WEIGHT_DECAY:
            # Apply different weight decay to Batchnorm and non-batchnorm parameters.
            p_bn = [p for n, p in model.named_parameters() if "bn" in n]
            p_non_bn = [p for n, p in model.named_parameters() if "bn" not in n]
            optim_params = [
                {"params": p_bn, "weight_decay": cfg.BN.CUSTOM_WEIGHT_DECAY},
                {"params": p_non_bn, "weight_decay": cfg.OPTIM.WEIGHT_DECAY},
            ]
        else:
            optim_params = model.parameters()
    # only optimize the affine parameters after normalization
    elif cfg.OPTIM.PARAMS == 'bn_related':
        bn_params = []

        for name, p in model.named_parameters():
            if "bn" in name:
                bn_params.append(p)
            else:
                p.requires_grad = False

        # Apply different weight decay to Batchnorm and non-batchnorm parameters.
        bn_weight_decay = (
            cfg.BN.CUSTOM_WEIGHT_DECAY
            if cfg.BN.USE_CUSTOM_WEIGHT_DECAY
            else cfg.OPTIM.WEIGHT_DECAY
        )
        optim_params = [
            {"params": bn_params, "weight_decay": bn_weight_decay},
        ]
    # only optimize gamma and beta after normalization
    elif cfg.OPTIM.PARAMS == 'gamma_beta':
        bn_params = []

        for name, p in model.named_parameters():
            if "bn" in name and ("gamma" in name or "beta" in name):
                bn_params.append(p)
            else:
                p.requires_grad = False

        # Apply different weight decay to Batchnorm and non-batchnorm parameters.
        bn_weight_decay = (
            cfg.BN.CUSTOM_WEIGHT_DECAY
            if cfg.BN.USE_CUSTOM_WEIGHT_DECAY
            else cfg.OPTIM.WEIGHT_DECAY
        )
        optim_params = [
            {"params": bn_params, "weight_decay": bn_weight_decay},
        ]
    # only optimize spade for FG1
    elif cfg.OPTIM.PARAMS == 'spade_related':
        spade_params = []

        for name, p in model.named_parameters():
            if "spade" in name:
                spade_params.append(p)
            else:
                p.requires_grad = False

        bn_weight_decay = (
            cfg.BN.CUSTOM_WEIGHT_DECAY
            if cfg.BN.USE_CUSTOM_WEIGHT_DECAY
            else cfg.OPTIM.WEIGHT_DECAY
        )
        optim_params = [
            {"params": spade_params, "weight_decay": bn_weight_decay},
        ]
    
    if cfg.OPTIM.METHOD == 'sgd':
        return torch.optim.SGD(
            optim_params,
            lr=cfg.OPTIM.BASE_LR,
            momentum=cfg.OPTIM.MOMENTUM,
            weight_decay=cfg.OPTIM.WEIGHT_DECAY,
            dampening=cfg.OPTIM.DAMPENING,
            nesterov=cfg.OPTIM.NESTEROV,
        )
    elif cfg.OPTIM.METHOD == 'adam':
        return torch.optim.Adam(
            optim_params,
            lr=cfg.OPTIM.BASE_LR,
            weight_decay=cfg.OPTIM.WEIGHT_DECAY,
        )


def lr_fun_steps(cur_epoch):
    """Steps schedule (cfg.OPTIM.LR_POLICY = 'steps')."""
    ind = [i for i, s in enumerate(cfg.OPTIM.STEPS) if cur_epoch >= s][-1]
    return cfg.OPTIM.BASE_LR * (cfg.OPTIM.LR_MULT ** ind)


def lr_fun_exp(cur_epoch):
    """Exponential schedule (cfg.OPTIM.LR_POLICY = 'exp')."""
    return cfg.OPTIM.BASE_LR * (cfg.OPTIM.GAMMA ** cur_epoch)


def lr_fun_cos(cur_epoch):
    """Cosine schedule (cfg.OPTIM.LR_POLICY = 'cos')."""
    base_lr, max_epoch = cfg.OPTIM.BASE_LR, cfg.OPTIM.MAX_EPOCH
    return 0.5 * base_lr * (1.0 + np.cos(np.pi * cur_epoch / max_epoch))


def get_lr_fun():
    """Retrieves the specified lr policy function"""
    lr_fun = "lr_fun_" + cfg.OPTIM.LR_POLICY
    if lr_fun not in globals():
        raise NotImplementedError("Unknown LR policy:" + cfg.OPTIM.LR_POLICY)
    return globals()[lr_fun]


def get_epoch_lr(cur_epoch):
    """Retrieves the lr for the given epoch according to the policy."""
    lr = get_lr_fun()(cur_epoch)
    # Linear warmup
    if cur_epoch < cfg.OPTIM.WARMUP_EPOCHS:
        alpha = cur_epoch / cfg.OPTIM.WARMUP_EPOCHS
        warmup_factor = cfg.OPTIM.WARMUP_FACTOR * (1.0 - alpha) + alpha
        lr *= warmup_factor
    return lr


def set_lr(optimizer, new_lr):
    """Sets the optimizer lr to the specified value."""
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr
