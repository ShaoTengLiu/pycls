#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""CIFAR10 dataset."""

import os
import pickle

import numpy as np
import pycls.core.logging as logging
import pycls.datasets.transforms as transforms
import torch.utils.data
from pycls.core.config import cfg

import torchvision.transforms as ttransforms

logger = logging.get_logger(__name__)

# Per-channel mean and SD values in BGR order
_MEAN_BGR = [125.3, 123.0, 113.9]
_SD_BGR = [63.0, 62.1, 66.7]

# RGB order, normalize to (0, 1)
_MEAN_RGB = [0.447, 0.482, 0.491]
_SD_RGB = [0.262, 0.244, 0.247]

def get_aug(aug_type):
    if aug_type == 'original':
        return ttransforms.Compose([
                ttransforms.RandomCrop(cfg.TRAIN.IM_SIZE, 
                                    padding=cfg.TRAIN.IM_SIZE // 8,
                                    padding_mode='reflect'),
                ttransforms.RandomHorizontalFlip(p=0.5), 
                ttransforms.ToTensor(),
                ttransforms.Normalize(_MEAN_RGB, _SD_RGB)])
    
    elif aug_type == 'test':
        return ttransforms.Compose([
                            ttransforms.ToTensor(),
                            ttransforms.Normalize(_MEAN_RGB, _SD_RGB)])

def get_npy(split, corruption, level, data_path):
    # np.load('data/CIFAR-10-C/train/%s_%d_images.npy' % (corruption, level))
    assert level in [1, 2, 3, 4, 5]
    assert corruption in ['gaussian_noise', 'shot_noise', 'impulse_noise', 
                          'defocus_blur', 'glass_blur', 'motion_blur', 
                          'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 
                          'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
    if split == 'train':
        len_npy = 50000 # num of training images
        npy_all = np.load(data_path + '/train/%s.npy' % corruption)
        return npy_all[(level-1)*len_npy:level*len_npy]
    else:
        len_npy = 10000 # num of training images
        npy_all = np.load(data_path + '/val/%s.npy' % corruption)
        return npy_all[(level-1)*len_npy:level*len_npy]

class Cifar100(torch.utils.data.Dataset):
    """CIFAR-100 dataset."""

    def __init__(self, data_path, split, corruption_type, corruption_level):
        assert os.path.exists(data_path), "Data path '{}' not found".format(data_path)
        assert split in ["train", "test"], "Split '{}' not supported for cifar".format(
            split
        )
        logger.info("Constructing CIFAR-100 {}...".format(split))
        self._data_path = data_path
        self._split = split
        self._corruption_type = corruption_type
        self._corruption_level = corruption_level
        # Data format:
        #   self._inputs - (split_size, 3, im_size, im_size) ndarray
        #   self._labels - split_size list
        self._inputs, self._labels = self._load_data()

        self.aug_type = cfg.AUG.TYPE # default to use CHW BGR version
        assert self.aug_type != 'default'
        if self._split == "train":
            self.img_aug = get_aug(cfg.AUG.TYPE)
        else:
            self.img_aug = ttransforms.Compose([
                            ttransforms.ToTensor(),
                            ttransforms.Normalize(_MEAN_RGB, _SD_RGB)])

    def _load_data(self):
        """Loads data in memory."""
        logger.info("{} data path: {}".format(self._split, self._data_path))
        # Compute data file path
        file_path = os.path.join(self._data_path, self._split)

        # Load data batches
        with open(file_path, "rb") as f:
            data_dict = pickle.load(f, encoding="bytes")
        inputs, labels = data_dict[b"data"], data_dict[b"fine_labels"]

        # reshape the inputs (N x 3072 --> N x 32 x 32 x 3)
        inputs = inputs.reshape((-1, 3, 32, 32)).astype(np.float32)
        inputs = inputs.transpose((0, 2, 3, 1)) # CHW --> HWC

        if self._corruption_type != 'original':
            inputs = get_npy(self._split, 
                             self._corruption_type, 
                             self._corruption_level, 
                             self._data_path.replace('cifar100', 'CIFAR-100-C'))

        # CORRUPTED DATA is already N x 32 x 32 x 3
        return inputs, labels

    def __getitem__(self, index):
        im_np, label = self._inputs[index, ...].copy(), self._labels[index]
        im_pil = Image.fromarray(im_np[:, :, ::-1].astype(np.uint8)) #BGR --> RGB
        return self.img_aug(im_pil), label

    def __len__(self):
        return self._inputs.shape[0]

class Cifar10(torch.utils.data.Dataset):
    """CIFAR-10 dataset."""

    def __init__(self, data_path, split, corruption_type, corruption_level):
        assert os.path.exists(data_path), "Data path '{}' not found".format(data_path)
        assert split in ["train", "test"], "Split '{}' not supported for cifar".format(
            split
        )
        logger.info("Constructing CIFAR-10 {}...".format(split))
        self._data_path = data_path
        self._split = split
        self._corruption_type = corruption_type
        self._corruption_level = corruption_level
        self._inputs, self._labels = self._load_data()

        if self._split == "train":
            self.img_aug = get_aug(cfg.AUG.TYPE)
        else:
            self.img_aug = ttransforms.Compose([
                            ttransforms.ToTensor(),
                            ttransforms.Normalize(_MEAN_RGB, _SD_RGB)])

    def _load_batch(self, batch_path):
        with open(batch_path, "rb") as f:
            d = pickle.load(f, encoding="bytes")
        return d[b"data"], d[b"labels"]

    def _load_data(self):
        """Loads data in memory."""
        logger.info("{} data path: {}".format(self._split, self._data_path))
        # Compute data batch names
        if self._split == "train":
            batch_names = ["data_batch_{}".format(i) for i in range(1, 6)]
        else:
            batch_names = ["test_batch"]
        # Load data batches
        inputs, labels = [], []
        for batch_name in batch_names:
            batch_path = os.path.join(self._data_path, batch_name)
            inputs_batch, labels_batch = self._load_batch(batch_path)
            inputs.append(inputs_batch)
            labels += labels_batch
        # Combine and reshape the inputs
        inputs = np.vstack(inputs) # N x 3072
        inputs = inputs.reshape((-1, 3, 32, 32)).astype(np.float32)
        inputs = inputs.transpose((0, 2, 3, 1)) # CHW --> HWC

        if self._corruption_type != 'original':
            if self._corruption_type == 'cifar10.1':
                assert self._split == 'test'
                dp = self._data_path.replace('cifar10', 'cifar10.1')
                inputs = np.load(dp + '/cifar10images.npy')
                labels = np.load(dp + '/cifar10labels.npy')
            else:
                inputs = get_npy(self._split, 
                            self._corruption_type, 
                            self._corruption_level, 
                            self._data_path.replace('cifar10', 'CIFAR-10-C'))

        # CORRUPTED DATA is already N x 32 x 32 x 3

        return inputs, labels

    def __getitem__(self, index):
        im_np, label = self._inputs[index, ...].copy(), self._labels[index]
        im_pil = Image.fromarray(im_np[:, :, ::-1].astype(np.uint8)) #BGR --> RGB
        return self.img_aug(im_pil), label

    def __len__(self):
        return self._inputs.shape[0]