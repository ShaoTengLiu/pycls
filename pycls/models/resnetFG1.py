#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""No input filter + gates"""
import torch.nn as nn
from torch.nn import functional as F

from pycls.core.config import cfg
from pycls.utils import get_norm
import pycls.core.net as net

import torch

# Stage depths for ImageNet models
_IN_STAGE_DS = {50: (3, 4, 6, 3), 101: (3, 4, 23, 3), 152: (3, 8, 36, 3)}

def get_spade_param(channels):
    name = cfg.RESNET.SPADE_VER
    if name == 'v1':
        param = nn.Parameter(torch.zeros(1))
    elif name == 'v2':
        param = nn.Parameter(torch.zeros(channels))
    return param

def get_feats_fun(w_in, w_out, stride):
    return nn.Sequential(
                    nn.Conv2d(w_in, w_out, kernel_size=3, stride=stride, padding=1, bias=False),
                    nn.ReLU(inplace=True),
                )

def get_spade_fun(w_in, w_out, stride):
    return nn.Conv2d(w_in, w_out * 2, kernel_size=1, stride=stride, padding=0, bias=False)

def get_trans_fun(name):
    """Retrieves the transformation function by name."""
    trans_funs = {
        "basic_transform": BasicTransform,
        "bottleneck_transform": BottleneckTransform,
    }
    assert (
        name in trans_funs.keys()
    ), "Transformation function '{}' not supported".format(name)
    return trans_funs[name]


class ResHead(nn.Module):
    """ResNet head."""

    def __init__(self, w_in, nc):
        super(ResHead, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(w_in, nc, bias=True)

    def forward(self, x):
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class BasicTransform(nn.Module):
    """Basic transformation: 3x3, 3x3"""

    def __init__(self, w_in, w_out, stride, w_b=None, num_gs=1):
        assert (
            w_b is None and num_gs == 1
        ), "Basic transform does not support w_b and num_gs options"
        super(BasicTransform, self).__init__()
        self._construct(w_in, w_out, stride)

    def _construct(self, w_in, w_out, stride):
        # 3x3, BN, ReLU
        self.a = nn.Conv2d(
            w_in, w_out, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.a_bn = get_norm(cfg.RESNET.NORM_FUNC, w_out)
        #nn.BatchNorm2d(w_out, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        self.a_relu = nn.ReLU(inplace=cfg.MEM.RELU_INPLACE)
        self.a_spade_g = get_spade_param(w_out)
        self.a_spade_b = get_spade_param(w_out)

        # 3x3, BN
        self.b = nn.Conv2d(w_out, w_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.b_bn = get_norm(cfg.RESNET.NORM_FUNC, w_out)
        # nn.BatchNorm2d(w_out, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        self.b_bn.final_bn = True
        self.b_spade_g = get_spade_param(w_out)
        self.b_spade_b = get_spade_param(w_out)

    def forward(self, x, g, b):
        # for layer in self.children():
        #     x = layer(x)
        x = self.a(x)
        x = self.a_bn(x)
        x = x * (1 + g * self.a_spade_g.reshape(1, -1, 1, 1)) + b * self.a_spade_b.reshape(1, -1, 1, 1)
        x = self.a_relu(x)

        x = self.b(x)
        x = self.b_bn(x)
        x = x * (1 + g * self.b_spade_g.reshape(1, -1, 1, 1)) + b * self.b_spade_b.reshape(1, -1, 1, 1)
        return x


class BottleneckTransform(nn.Module):
    """Bottleneck transformation: 1x1, 3x3, 1x1"""

    def __init__(self, w_in, w_out, stride, w_b, num_gs):
        super(BottleneckTransform, self).__init__()
        self._construct(w_in, w_out, stride, w_b, num_gs)

    def _construct(self, w_in, w_out, stride, w_b, num_gs):
        # MSRA -> stride=2 is on 1x1; TH/C2 -> stride=2 is on 3x3
        (str1x1, str3x3) = (stride, 1) if cfg.RESNET.STRIDE_1X1 else (1, stride)
        # 1x1, BN, ReLU
        self.a = nn.Conv2d(
            w_in, w_b, kernel_size=1, stride=str1x1, padding=0, bias=False
        )
        self.a_bn = get_norm(cfg.RESNET.NORM_FUNC, w_b)
        # nn.BatchNorm2d(w_b, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        self.a_relu = nn.ReLU(inplace=cfg.MEM.RELU_INPLACE)
        # 3x3, BN, ReLU
        self.b = nn.Conv2d(
            w_b, w_b, kernel_size=3, stride=str3x3, padding=1, groups=num_gs, bias=False
        )
        self.b_bn = get_norm(cfg.RESNET.NORM_FUNC, w_b)
        # nn.BatchNorm2d(w_b, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        self.b_relu = nn.ReLU(inplace=cfg.MEM.RELU_INPLACE)
        # 1x1, BN
        self.c = nn.Conv2d(w_b, w_out, kernel_size=1, stride=1, padding=0, bias=False)
        self.c_bn = get_norm(cfg.RESNET.NORM_FUNC, w_out)
        # nn.BatchNorm2d(w_out, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        self.c_bn.final_bn = True

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class ResBlock(nn.Module):
    """Residual block: x + F(x)"""

    def __init__(self, w_in, w_out, stride, trans_fun, w_b=None, num_gs=1):
        super(ResBlock, self).__init__()
        self._construct(w_in, w_out, stride, trans_fun, w_b, num_gs)

    def _add_skip_proj(self, w_in, w_out, stride):
        self.proj = nn.Conv2d(
            w_in, w_out, kernel_size=1, stride=stride, padding=0, bias=False
        )
        self.spade_g = get_spade_param(w_out)
        self.spade_b = get_spade_param(w_out)
        self.bn = get_norm(cfg.RESNET.NORM_FUNC, w_out)
        # nn.BatchNorm2d(w_out, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)

    def _construct(self, w_in, w_out, stride, trans_fun, w_b, num_gs):
        # Use skip connection with projection if shape changes
        self.proj_block = (w_in != w_out) or (stride != 1)
        if self.proj_block:
            self._add_skip_proj(w_in, w_out, stride)
        self.f = trans_fun(w_in, w_out, stride, w_b, num_gs)
        self.relu = nn.ReLU(cfg.MEM.RELU_INPLACE)

    def forward(self, x, g, b):
        if self.proj_block:
            x = self.bn(self.proj(x)) * (1 + g * self.spade_g.reshape(1, -1, 1, 1)) + b * self.spade_b.reshape(1, -1, 1, 1) + self.f(x, g, b)
            # x = self.bn(self.proj(x)) + self.f(x)
        else:
            # x = x + self.f(x)
            x = x + self.f(x, g, b)
        x = self.relu(x)
        return x


class ResStage(nn.Module):
    """Stage of ResNet."""

    def __init__(self, w_in, w_out, stride, d, w_b=None, num_gs=1):
        super(ResStage, self).__init__()
        self._construct(w_in, w_out, stride, d, w_b, num_gs)

    def _construct(self, w_in, w_out, stride, d, w_b, num_gs):
        # Construct the blocks
        for i in range(d):
            # Stride and w_in apply to the first block of the stage
            b_stride = stride if i == 0 else 1
            b_w_in = w_in if i == 0 else w_out
            # Retrieve the transformation function
            trans_fun = get_trans_fun(cfg.RESNET.TRANS_FUN)
            # Construct the block
            res_block = ResBlock(b_w_in, w_out, b_stride, trans_fun, w_b, num_gs)
            self.add_module("b{}".format(i + 1), res_block)

    def forward(self, x, g, b):
        for block in self.children():
            x = block(x, g, b)
        return x


class ResStem(nn.Module):
    """Stem of ResNet."""

    def __init__(self, w_in, w_out):
        super(ResStem, self).__init__()
        if "cifar" in cfg.TRAIN.DATASET:
            self._construct_cifar(w_in, w_out)
        else:
            self._construct_imagenet(w_in, w_out)

    def _construct_cifar(self, w_in, w_out):
        # 3x3, BN, ReLU
        self.conv = nn.Conv2d(
            w_in, w_out, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn = get_norm(cfg.RESNET.NORM_FUNC, w_out)
        #nn.BatchNorm2d(w_out, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        self.relu = nn.ReLU(cfg.MEM.RELU_INPLACE)

        self.spade_g = get_spade_param(w_out)
        self.spade_b = get_spade_param(w_out)

    def _construct_imagenet(self, w_in, w_out):
        # 7x7, BN, ReLU, maxpool
        self.conv = nn.Conv2d(
            w_in, w_out, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn = get_norm(cfg.RESNET.NORM_FUNC, w_out)
        # nn.BatchNorm2d(w_out, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        self.relu = nn.ReLU(cfg.MEM.RELU_INPLACE)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x, g, b):
        # for layer in self.children():
        #     x = layer(x)
        x = self.conv(x)
        x = self.bn(x)
        x = x * (1 + g * self.spade_g.reshape(1, -1, 1, 1)) + b * self.spade_b.reshape(1, -1, 1, 1)
        x = self.relu(x)
        return x


class ResNetFG1(nn.Module):
    """ResNet model."""

    def __init__(self):
        assert cfg.TRAIN.DATASET in [
            "svhn",
            "mnist",
            "mnistm",
            "cifar10",
            "cifar100",
            "imagenet",
        ], "Training ResNet on {} is not supported".format(cfg.TRAIN.DATASET)
        assert cfg.TEST.DATASET in [
            "svhn",
            "mnist",
            "mnistm",
            "cifar10",
            "cifar100",
            "imagenet",
        ], "Testing ResNet on {} is not supported".format(cfg.TEST.DATASET)
        super(ResNetFG1, self).__init__()
        if "imagenet" in cfg.TRAIN.DATASET:
            self._construct_imagenet()
        else:
            self._construct_cifar()
        self.apply(net.init_weights)

    def _construct_cifar(self):
        assert (
            cfg.MODEL.DEPTH - 2
        ) % 6 == 0, "Model depth should be of the format 6n + 2 for cifar"
        # logger.info("Constructing: ResNet-{}-w{}".format(cfg.MODEL.DEPTH, cfg.MODEL.WIDTH))
        # Each stage has the same number of blocks for cifar
        d, w = int((cfg.MODEL.DEPTH - 2) / 6), cfg.MODEL.WIDTH
        # Stem: (N, 3, 32, 32) -> (N, 16, 32, 32)
        self.stem_spade_f = get_feats_fun(3, 16 * w, 1)
        self.stem_spade_h = get_spade_fun(16 * w, 16 * w, 1)
        self.stem = ResStem(w_in=3, w_out=16 * w)
        # self.s0 = ResStem(w_in=3, w_out=16 * w)

        # Stage 1: (N, 16, 32, 32) -> (N, 16, 32, 32)
        self.s1_spade_f = get_feats_fun(16 * w, 16 * w, 1)
        self.s1_spade_h = get_spade_fun(16 * w, 16 * w, 1)
        self.s1 = ResStage(w_in=16 * w, w_out=16 * w, stride=1, d=d)

        # Stage 2: (N, 16, 32, 32) -> (N, 32, 16, 16)
        self.s2_spade_f = get_feats_fun(16 * w, 32 * w, 2)
        self.s2_spade_h = get_spade_fun(32 * w, 32 * w, 1)
        self.s2 = ResStage(w_in=16 * w, w_out=32 * w, stride=2, d=d)

        # Stage 3: (N, 32, 16, 16) -> (N, 64, 8, 8)
        self.s3_spade_f = get_feats_fun(32 * w, 64 * w, 2)
        self.s3_spade_h = get_spade_fun(64 * w, 64 * w, 1)
        self.s3 = ResStage(w_in=32 * w, w_out=64 * w, stride=2, d=d)
        # Head: (N, 64, 8, 8) -> (N, num_classes)
        self.head = ResHead(w_in=64 * w, nc=cfg.MODEL.NUM_CLASSES)

    def _construct_imagenet(self):
        # Retrieve the number of blocks per stage
        (d1, d2, d3, d4) = _IN_STAGE_DS[cfg.MODEL.DEPTH]
        # Compute the initial bottleneck width
        num_gs = cfg.RESNET.NUM_GROUPS
        w_b = cfg.RESNET.WIDTH_PER_GROUP * num_gs
        # Stem: (N, 3, 224, 224) -> (N, 64, 56, 56)
        self.stem = ResStem(w_in=3, w_out=64)
        # self.s0 = ResStem(w_in=3, w_out=64)
        # Stage 1: (N, 64, 56, 56) -> (N, 256, 56, 56)
        self.s1 = ResStage(w_in=64, w_out=256, stride=1, d=d1, w_b=w_b, num_gs=num_gs)
        # Stage 2: (N, 256, 56, 56) -> (N, 512, 28, 28)
        self.s2 = ResStage(
            w_in=256, w_out=512, stride=2, d=d2, w_b=w_b * 2, num_gs=num_gs
        )
        # Stage 3: (N, 512, 56, 56) -> (N, 1024, 14, 14)
        self.s3 = ResStage(
            w_in=512, w_out=1024, stride=2, d=d3, w_b=w_b * 4, num_gs=num_gs
        )
        # Stage 4: (N, 1024, 14, 14) -> (N, 2048, 7, 7)
        self.s4 = ResStage(
            w_in=1024, w_out=2048, stride=2, d=d4, w_b=w_b * 8, num_gs=num_gs
        )
        # Head: (N, 2048, 7, 7) -> (N, num_classes)
        self.head = ResHead(w_in=2048, nc=cfg.MODEL.NUM_CLASSES)

    def forward(self, x):
        # for module in self.children():
        #     x = module(x)

        y = self.stem_spade_f(x)
        s = self.stem_spade_h(y)
        g, b = s.chunk(2, 1)
        x = self.stem(x, g, b)
        
        y = self.s1_spade_f(y)
        s = self.s1_spade_h(y)
        g, b = s.chunk(2, 1)
        x = self.s1(x, g, b)

        y = self.s2_spade_f(y)
        s = self.s2_spade_h(y)
        g, b = s.chunk(2, 1)
        x = self.s2(x, g, b)

        y = self.s3_spade_f(y)
        s = self.s3_spade_h(y)
        g, b = s.chunk(2, 1)
        x = self.s3(x, g, b)

        x = self.head(x)
        return x