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

