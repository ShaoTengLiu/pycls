# FTTA

**FTTA** is the official code for paper [Fully Test-time Adaptation by Entropy Minimization](https://arxiv.org/abs/1905.13214). **FTTA** is developed based on [pycls](https://github.com/facebookresearch/pycls).

<!-- <div align="center">
  <img src="docs/regnetx_nets.png" width="550px" />
  <p align="center"><b>pycls</b> provides a large set of baseline models across a wide range of flop regimes.</p>
</div> -->

## Introduction

**FTTA** is ......

## Using FTTA

Please see [`INSTALL.md`](docs/INSTALL.md) for brief installation instructions. After installation, please see [`GETTING_STARTED.md`](docs/GETTING_STARTED.md) for basic instructions and example commands on training and evaluation with **pycls**.

Training on CIFAR with 1 GPU: 

```
python tools/train_net.py \
    --cfg configs/archive/cifar/resnet/R-56_nds_1gpu.yaml \
    OUT_DIR /tmp
```

Use entropy feedback to adapt on CIFAR with 1 GPU: 

```
python tools/test_feedback_net.py \
    --cfg configs/archive/cifar10/R-26_w4_1gpu_adaptbn_entropy_precisebn.yaml
```

<!-- ## Model Zoo

We provide a large set of baseline results and pretrained models available for download in the **pycls** [Model Zoo](MODEL_ZOO.md); including the simple, fast, and effective [RegNet](https://arxiv.org/abs/2003.13678) models that we hope can serve as solid baselines across a wide range of flop regimes. -->

## Citing FTTA

If you find **FTTA** helpful in your research or refer to the baseline results in the [Model Zoo](MODEL_ZOO.md), please consider citing:

```
@InProceedings{Radosavovic2019,
  title = {On Network Design Spaces for Visual Recognition},
  author = {Radosavovic, Ilija and Johnson, Justin and Xie, Saining and Lo, Wan-Yen and Doll{\'a}r, Piotr},
  booktitle = {ICCV},
  year = {2019}
}

@InProceedings{Radosavovic2020,
  title = {Designing Network Design Spaces},
  author = {Radosavovic, Ilija and Kosaraju, Raj Prateek and Girshick, Ross and He, Kaiming and Doll{\'a}r, Piotr},
  booktitle = {CVPR},
  year = {2020}
}
```

<!-- ## License

**pycls** is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information. -->

## Contributing

We actively welcome your pull requests! Please see [`CONTRIBUTING.md`](docs/CONTRIBUTING.md) and [`CODE_OF_CONDUCT.md`](docs/CODE_OF_CONDUCT.md) for more info.
