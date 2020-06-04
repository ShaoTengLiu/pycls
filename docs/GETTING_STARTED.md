# Getting Started

This document provides basic instructions for training and evaluation using **pycls**.

- For general information about **pycls**, please see [`README.md`](../README.md)
- For installation instructions, please see [`INSTALL.md`](INSTALL.md)

## Training Models

Training on CIFAR10 with 1 GPU: 

```bash
python tools/train_net.py \
    --cfg configs/cifar10/R-26_w4_1gpu_cifar10_bn.yaml
```

Training on ImageNet with 8 GPUs:

```bash
python tools/train_net.py \
    --cfg configs/imagenet/R-50-1x64d_dds_8gpu.yaml
```

## Finetuning Models

Finetuning on ImageNet with 8 GPU:

```bash
python tools/train_net.py \
    ---cfg configs/imagenet/R-50-1x64d_dds_8gpu.yaml \
    TRAIN.WEIGHTS /path/to/weights/file
```

## Evaluating Models

Evaluation on ImageNet:

```bash
python tools/test_net.py \
    --cfg configs/imagenet/R-50-1x64d_dds_8gpu.yaml \
    TEST.WEIGHTS /path/to/weights/file
```

## Adapting Models

Fully test-time adaptation on CIFAR10:

```bash
python tools/test_net_ftta.py \
    --cfg configs/cifar10/R-26_w4_1gpu_adaptbn_entropy.yaml
```

## 