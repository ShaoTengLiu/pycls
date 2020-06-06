#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Optimizer."""

import numpy as np
import torch
from pycls.core.config import cfg


def show_parameters(model):
    """Show the parameters of the model."""
    if cfg.OPTIM.PARAMS == 'default':
        if cfg.BN.USE_CUSTOM_WEIGHT_DECAY:
            # Apply different weight decay to Batchnorm and non-batchnorm parameters.
            p_bn = [p for n, p in model.named_parameters() if "bn" in n]
            p_non_bn = [p for n, p in model.named_parameters() if "bn" not in n]
            params = [
                {"params": p_bn, "weight_decay": cfg.BN.CUSTOM_WEIGHT_DECAY},
                {"params": p_non_bn, "weight_decay": cfg.OPTIM.WEIGHT_DECAY},
            ]
        else:
            params = model.parameters()
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
        params = [
            {"params": bn_params, "weight_decay": bn_weight_decay},
        ]
    # only optimize gamma and beta after normalization
    elif cfg.OPTIM.PARAMS == 'gamma_beta':
        bn_params = []

        for name, p in model.named_parameters():
            if "bn" in name and ("gamma" in name or "beta" in name):
                bn_params.append((name, p))
            else:
                p.requires_grad = False

        # Apply different weight decay to Batchnorm and non-batchnorm parameters.
        bn_weight_decay = (
            cfg.BN.CUSTOM_WEIGHT_DECAY
            if cfg.BN.USE_CUSTOM_WEIGHT_DECAY
            else cfg.OPTIM.WEIGHT_DECAY
        )
        params = [
            {"params": bn_params, "weight_decay": bn_weight_decay},
        ]
    print(params)
    return