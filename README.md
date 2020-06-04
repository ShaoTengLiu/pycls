# **Fully Test-time Adaptation by Entropy Minimization**

**FTTA** is the official code for paper **Fully Test-time Adaptation by Entropy Minimization**, based on [pycls](https://github.com/facebookresearch/pycls).

## Abstract

Faced with new and different data during testing, a model must adapt itself. We consider the setting of fully test-time adaptation, in which a supervised model confronts unlabeled test data from a different distribution, without the help of its labeled training data.  We propose an entropy minimization approach for adaptation: we take the model's confidence as our objective as measured by the entropy of its predictions.  During testing, we adapt the model by modulating its representation with affine transformations to minimize entropy.  Our experiments show improved robustness to corruptions for image classification on CIFAR-10/100 and ILSVRC and demonstrate the feasibility of target-only domain adaptation for digit classification on MNIST and SVHN.

## Use FTTA

Please see [`INSTALL.md`](docs/INSTALL.md) for brief installation instructions of **pycls**. After installation, please see [`GETTING_STARTED.md`](docs/GETTING_STARTED.md) for basic instructions and example commands on training and evaluation.

Train image classification ConvNet with a supervised loss on CIFAR100 with 1 GPU: 

```bash
python tools/train_net.py \
    --cfg configs/archive/cifar/resnet/R-56_nds_1gpu.yaml \
    OUT_DIR /tmp
```

Fully test-time adaptation via entropy minimization on CIFAR100 with 1 GPU: 

```bash
python tools/test_feedback_net.py \
    --cfg configs/archive/cifar10/R-26_w4_1gpu_adaptbn_entropy_precisebn.yaml
```

## Citing FTTA

If you find **FTTA** helpful in your research, please consider citing:

```latex

```

<!-- ## License

**FTTA** is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information. -->

