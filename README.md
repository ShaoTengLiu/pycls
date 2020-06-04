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

## Results

### cifar10

**level5**

| Model           | gauss | shot  | impulse | defocus | glass | motion | zoom  | snow  | frost | fog   | bright | contrast | elastic | pixelate | jpeg  |
| --------------- | ----- | ----- | ------- | ------- | ----- | ------ | ----- | ----- | ----- | ----- | ------ | -------- | ------- | -------- | ----- |
| Baseline        | 67.2  | 59.59 | 62.48   | 42.91   | 47.79 | 32.7   | 36.39 | 22.96 | 35.47 | 28.81 | 7.69   | 62.39    | 24.82   | 52.09    | 28.53 |
| Bn_update       | 25.35 | 22.17 | 32.63   | 10.75   | 31.19 | 10.77  | 9.94  | 13.9  | 14.17 | 11.07 | 6.19   | 10.01    | 18.54   | 17.72    | 25.53 |
| adaptbn_entropy | 18.91 | 16.79 | 25.19   | 9.22    | 27.73 | 11.07  | 9.13  | 12.51 | 12.49 | 10.14 | 6.64   | 7.87     | 17.88   | 11.99    | 19.9  |

**level4**

| Model           | gauss | shot  | impulse | defocus | glass | motion | zoom  | snow  | frost | fog  | bright | contrast | elastic | pixelate | jpeg  |
| --------------- | ----- | ----- | ------- | ------- | ----- | ------ | ----- | ----- | ----- | ---- | ------ | -------- | ------- | -------- | ----- |
| Baseline        | 63.2  | 48.52 | 49.92   | 19.74   | 50.8  | 23.86  | 26.37 | 17.18 | 25.18 | 10.9 | 5.73   | 20.37    | 19.97   | 31.87    | 23.52 |
| Bn_update       | 23.11 | 17.59 | 25.82   | 6.33    | 31.19 | 8.66   | 7.56  | 12.87 | 11.36 | 6.17 | 5.03   | 6.38     | 11.68   | 11.0     | 20.83 |
| adaptbn_entropy | 18.1  | 15.25 | 20.69   | 6.53    | 24.23 | 8.43   | 7.25  | 12.09 | 10.82 | 6.52 | 5.15   | 6.93     | 11.91   | 9.01     | 18.19 |

**level3**

| Model           | gauss | shot  | impulse | defocus | glass | motion | zoom  | snow  | frost | fog  | bright | contrast | elastic | pixelate | jpeg  |
| --------------- | ----- | ----- | ------- | ------- | ----- | ------ | ----- | ----- | ----- | ---- | ------ | -------- | ------- | -------- | ----- |
| Baseline        | 56.86 | 42.22 | 36.63   | 10.56   | 38.91 | 24.1   | 20.47 | 14.46 | 23.04 | 7.2  | 4.97   | 11.32    | 13.06   | 14.7     | 20.46 |
| Bn_update       | 19.58 | 15.22 | 17.7    | 4.67    | 20.64 | 8.98   | 6.75  | 11.61 | 10.88 | 5.17 | 4.62   | 5.57     | 7.35    | 7.78     | 18.39 |
| adaptbn_entropy | 15.71 | 12.83 | 14.74   | 4.99    | 16.38 | 8.32   | 6.75  | 11.56 | 10.29 | 6.15 | 4.96   | 5.48     | 8.02    | 7.89     | 15.42 |

**level2**

| Model           | gauss | shot  | impulse | defocus | glass | motion | zoom  | snow  | frost | fog  | bright | contrast | elastic | pixelate | jpeg  |
| --------------- | ----- | ----- | ------- | ------- | ----- | ------ | ----- | ----- | ----- | ---- | ------ | -------- | ------- | -------- | ----- |
| Baseline        | 40.61 | 22.67 | 26.96   | 5.72    | 41.38 | 15.53  | 14.67 | 18.69 | 13.94 | 5.5  | 4.36   | 7.8      | 8.85    | 10.23    | 18.25 |
| Bn_update       | 13.23 | 9.12  | 13.03   | 4.23    | 21.08 | 6.78   | 5.79  | 11.5  | 8.17  | 4.56 | 4.28   | 5.03     | 6.66    | 6.59     | 16.87 |
| adaptbn_entropy | 11.39 | 8.56  | 12.15   | 4.81    | 16.03 | 7.2    | 6.12  | 9.4   | 8.48  | 5.32 | 5.26   | 5.19     | 7.15    | 6.66     | 13.59 |

**level1**

| Model           | gauss | shot  | impulse | defocus | glass | motion | zoom | snow | frost | fog  | bright | contrast | elastic | pixelate | jpeg  |
| --------------- | ----- | ----- | ------- | ------- | ----- | ------ | ---- | ---- | ----- | ---- | ------ | -------- | ------- | -------- | ----- |
| Baseline        | 21.1  | 13.32 | 14.64   | 4.36    | 42.88 | 8.97   | 11.6 | 8.32 | 8.66  | 4.43 | 4.15   | 4.74     | 8.59    | 5.88     | 12.29 |
| Bn_update       | 9.41  | 7.09  | 8.95    | 4.03    | 21.48 | 5.55   | 5.87 | 7.14 | 6.39  | 4.25 | 4.02   | 4.18     | 6.96    | 5.49     | 10.87 |
| adaptbn_entropy | 9.27  | 7.26  | 8.19    | 4.71    | 17.37 | 6.06   | 6.01 | 7.71 | 6.64  | 4.93 | 4.84   | 4.75     | 7.7     | 6.14     | 10.78 |





### cifar100

**level5**

| Model           | gauss | shot  | impulse | defocus | glass | motion | zoom  | snow  | frost | fog   | bright | contrast | elastic | pixelate | jpeg  |
| --------------- | ----- | ----- | ------- | ------- | ----- | ------ | ----- | ----- | ----- | ----- | ------ | -------- | ------- | -------- | ----- |
| Baseline        | 89.89 | 88.05 | 95.1    | 65.62   | 81.44 | 55.63  | 58.96 | 52.99 | 65.9  | 59.11 | 31.85  | 77.76    | 51.43   | 76.49    | 57.59 |
| Bn_update       | 55.79 | 53.84 | 62.38   | 32.73   | 57.04 | 33.6   | 31.37 | 39.71 | 39.97 | 38.24 | 26.38  | 31.85    | 42.86   | 39.51    | 54.59 |
| adaptbn_entropy | 46.61 | 44.82 | 53.19   | 29.42   | 50.06 | 31.04  | 28.83 | 35.95 | 36.46 | 32.96 | 26.6   | 27.38    | 40.77   | 33.36    | 45.89 |

**level4**

| Model           | gauss | shot  | impulse | defocus | glass | motion | zoom  | snow  | frost | fog   | bright | contrast | elastic | pixelate | jpeg  |
| --------------- | ----- | ----- | ------- | ------- | ----- | ------ | ----- | ----- | ----- | ----- | ------ | -------- | ------- | -------- | ----- |
| Baseline        | 87.44 | 82.82 | 91.44   | 46.9    | 84.73 | 47.97  | 51.4  | 44.41 | 56.08 | 36.15 | 25.51  | 46.82    | 44.34   | 60.05    | 52.99 |
| Bn_update       | 51.59 | 47.38 | 54.47   | 25.22   | 57.38 | 29.96  | 28.01 | 36.59 | 35.77 | 28.18 | 23.3   | 25.28    | 34.3    | 31.62    | 50.18 |
| adaptbn_entropy | 43.89 | 39.56 | 46.29   | 25.09   | 50.49 | 28.55  | 26.94 | 34.71 | 33.19 | 27.03 | 23.95  | 23.74    | 33.48   | 28.77    | 41.5  |

**level3**

| Model           | gauss | shot  | impulse | defocus | glass | motion | zoom  | snow  | frost | fog   | bright | contrast | elastic | pixelate | jpeg  |
| --------------- | ----- | ----- | ------- | ------- | ----- | ------ | ----- | ----- | ----- | ----- | ------ | -------- | ------- | -------- | ----- |
| Baseline        | 84.44 | 77.47 | 79.93   | 34.84   | 74.77 | 48.33  | 45.34 | 41.39 | 54.72 | 28.97 | 23.27  | 35.31    | 36.81   | 39.75    | 49.24 |
| Bn_update       | 48.37 | 44.36 | 44.14   | 22.21   | 46.6  | 30.73  | 25.67 | 34.35 | 35.16 | 24.12 | 22.26  | 23.67    | 26.91   | 27.16    | 46.79 |
| adaptbn_entropy | 40.52 | 37.07 | 37.81   | 22.44   | 39.56 | 28.79  | 25.23 | 32.16 | 33.13 | 24.22 | 23.01  | 23.0     | 27.36   | 26.33    | 39.7  |

**level2**

| Model           | gauss | shot  | impulse | defocus | glass | motion | zoom  | snow  | frost | fog   | bright | contrast | elastic | pixelate | jpeg  |
| --------------- | ----- | ----- | ------- | ------- | ----- | ------ | ----- | ----- | ----- | ----- | ------ | -------- | ------- | -------- | ----- |
| Baseline        | 74.78 | 57.91 | 67.05   | 25.13   | 78.79 | 39.13  | 38.58 | 47.73 | 42.55 | 25.2  | 21.97  | 28.71    | 30.03   | 33.16    | 46.62 |
| Bn_update       | 40.55 | 34.1  | 38.67   | 21.38   | 46.36 | 26.84  | 24.14 | 34.57 | 30.59 | 22.57 | 21.51  | 22.31    | 25.24   | 25.42    | 43.55 |
| adaptbn_entropy | 35.36 | 30.51 | 34.38   | 21.65   | 39.2  | 26.46  | 23.92 | 31.37 | 30.0  | 23.01 | 22.01  | 22.48    | 26.01   | 25.56    | 37.4  |

**level1**

| Model           | gauss | shot  | impulse | defocus | glass | motion | zoom  | snow  | frost | fog   | bright | contrast | elastic | pixelate | jpeg  |
| --------------- | ----- | ----- | ------- | ------- | ----- | ------ | ----- | ----- | ----- | ----- | ------ | -------- | ------- | -------- | ----- |
| Baseline        | 56.59 | 44.41 | 44.56   | 21.5    | 79.15 | 30.49  | 33.77 | 29.99 | 32.86 | 21.93 | 21.37  | 22.3     | 9.5     | 25.66    | 37.92 |
| Bn_update       | 32.56 | 28.69 | 30.79   | 20.95   | 45.55 | 24.35  | 23.52 | 26.56 | 26.27 | 21.07 | 21.12  | 21.14    | 26.23   | 23.9     | 35.75 |
| adaptbn_entropy | 29.48 | 27.78 | 28.67   | 21.51   | 38.96 | 24.32  | 23.58 | 26.12 | 26.87 | 21.54 | 21.62  | 21.58    | 26.48   | 23.91    | 31.72 |


## Citing FTTA

If you find **FTTA** helpful in your research, please consider citing:

```latex

```

<!-- ## License

**FTTA** is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information. -->

