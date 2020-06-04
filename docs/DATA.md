# Setting Up Data Paths

Expected datasets structure for ImageNet:

```bash
imagenet
|_ train
|  |_ n01440764
|  |_ ...
|  |_ n15075141
|_ val
|  |_ n01440764
|  |_ ...
|  |_ n15075141
|_ ...
```

Expected datasets structure for CIFAR-10:

```bash
cifar10
|_ data_batch_1
|_ data_batch_2
|_ data_batch_3
|_ data_batch_4
|_ data_batch_5
|_ test_batch
|_ ...
```

Create a directory containing symlinks:

```bash
mkdir -p /path/pycls/pycls/datasets/data
```

Symlink ImageNet:

```bash
ln -s /path/imagenet /path/pycls/pycls/datasets/data/imagenet
```

Symlink CIFAR-10:

```bash
ln -s /path/cifar10 /path/pycls/pycls/datasets/data/cifar10
```

Symlink CIFAR-100:

```bash
ln -s /path/cifar100 /path/pycls/pycls/datasets/data/cifar100
```

Symlink CIFAR-10-C:

```bash
ln -s /path/cifar10_c /path/pycls/pycls/datasets/data/cifar10_c
```

Symlink CIFAR-100:

```bash
ln -s /path/cifar100_c /path/pycls/pycls/datasets/data/cifar100_c
```

Download the corrupted datasets ([cifar10_c](https://zenodo.org/record/2535967), [cifar100_c](https://zenodo.org/record/3555552), [imagenet_c](https://zenodo.org/record/2235448)) and put them in the data folder: 

(credit to: [Benchmarking Neural Network Robustness to Common Corruptions and Perturbations](https://github.com/hendrycks/robustness))

Expected datasets structure for CIFAR-10-C

```bash
cifar10_c
├── train
│   ├── brightness.npy
│   ├── contrast.npy
│   ├── defocus_blur.npy
│   ├── elastic_transform.npy
│   ├── fog.npy
│   ├── frost.npy
│   ├── gaussian_blur.npy
│   ├── gaussian_noise.npy
│   ├── glass_blur.npy
│   ├── impulse_noise.npy
│   ├── jpeg_compression.npy
│   ├── labels.npy
│   ├── motion_blur.npy
│   ├── pixelate.npy
│   ├── saturate.npy
│   ├── shot_noise.npy
│   ├── snow.npy
│   ├── spatter.npy
│   ├── speckle_noise.npy
│   └── zoom_blur.npy
└── val
    ├── brightness.npy
    ├── contrast.npy
    ├── defocus_blur.npy
    ├── elastic_transform.npy
    ├── fog.npy
    ├── frost.npy
    ├── gaussian_blur.npy
    ├── gaussian_noise.npy
    ├── glass_blur.npy
    ├── impulse_noise.npy
    ├── jpeg_compression.npy
    ├── labels.npy
    ├── motion_blur.npy
    ├── pixelate.npy
    ├── saturate.npy
    ├── shot_noise.npy
    ├── snow.npy
    ├── spatter.npy
    ├── speckle_noise.npy
    └── zoom_blur.npy
```

Expected datasets structure for CIFAR-100-C

```bash
cifar100_c
├── train
│   ├── brightness.npy
│   ├── contrast.npy
│   ├── defocus_blur.npy
│   ├── elastic_transform.npy
│   ├── fog.npy
│   ├── frost.npy
│   ├── gaussian_blur.npy
│   ├── gaussian_noise.npy
│   ├── glass_blur.npy
│   ├── impulse_noise.npy
│   ├── jpeg_compression.npy
│   ├── labels.npy
│   ├── motion_blur.npy
│   ├── pixelate.npy
│   ├── saturate.npy
│   ├── shot_noise.npy
│   ├── snow.npy
│   ├── spatter.npy
│   ├── speckle_noise.npy
│   └── zoom_blur.npy
└── val
    ├── brightness.npy
    ├── contrast.npy
    ├── defocus_blur.npy
    ├── elastic_transform.npy
    ├── fog.npy
    ├── frost.npy
    ├── gaussian_blur.npy
    ├── gaussian_noise.npy
    ├── glass_blur.npy
    ├── impulse_noise.npy
    ├── jpeg_compression.npy
    ├── labels.npy
    ├── motion_blur.npy
    ├── pixelate.npy
    ├── saturate.npy
    ├── shot_noise.npy
    ├── snow.npy
    ├── spatter.npy
    ├── speckle_noise.npy
    └── zoom_blur.npy
```

Expected datasets structure for imagenet_c:

```bash
imagenet_c
├── brightness
│   ├── 1
│   │   ├── n01440764
│   │   │   ├── ILSVRC2012_val_00000293.JPEG
│   │   │   ├── ...
│   │   ├── n01443537
│   │   │   ├── ILSVRC2012_val_00000236.JPEG
│   │   │   ├── ...
│   │   ├── ...
│   ├── ...
├── ...
```

