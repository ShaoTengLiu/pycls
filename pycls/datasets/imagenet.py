#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""ImageNet dataset."""

import os
import re

import cv2
import numpy as np
import pycls.core.logging as logging
import pycls.datasets.transforms as transforms
import torch.utils.data
from pycls.core.config import cfg

import torchvision.transforms as ttransforms
from PIL import Image, ImageFilter

logger = logging.get_logger(__name__)

# Per-channel mean and SD values in BGR order
_MEAN_BGR = [0.406, 0.456, 0.485]
_SD_BGR = [0.225, 0.224, 0.229]

_MEAN_RGB = [0.485, 0.456, 0.406]
_SD_RGB = [0.229, 0.224, 0.225]

# Eig vals and vecs of the cov mat
_EIG_VALS = np.array([[0.2175, 0.0188, 0.0045]])
_EIG_VECS = np.array(
    [[-0.5675, 0.7192, 0.4009], [-0.5808, -0.0045, -0.8140], [-0.5836, -0.6948, 0.4203]]
)
_EIG_VALS = torch.from_numpy(_EIG_VALS).float()
_EIG_VECS = torch.from_numpy(_EIG_VECS).float()

def get_aug(aug_type):
    if aug_type == 'original':
        return ttransforms.Compose([
                ttransforms.RandomResizedCrop(cfg.TRAIN.IM_SIZE),
                ttransforms.RandomHorizontalFlip(p=0.5),
                ttransforms.ToTensor(),
                Lighting(0.1, _EIG_VALS, _EIG_VECS),
                ttransforms.Normalize(_MEAN_RGB, _SD_RGB)])
    
    elif aug_type == 'test':
        return ttransforms.Compose([
                ttransforms.Resize(cfg.TEST.IM_SIZE),
                ttransforms.CenterCrop(cfg.TRAIN.IM_SIZE),
                ttransforms.ToTensor(),
                ttransforms.Normalize(_MEAN_RGB, _SD_RGB)])

def get_npy(split, corruption, level, data_path):
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

# Lighting data augmentation take from here - https://github.com/eladhoffer/convNet.pytorch/blob/master/preprocess.py
class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone()\
            .mul(alpha.view(1, 3).expand(3, 3))\
            .mul(self.eigval.view(1, 3).expand(3, 3))\
            .sum(1).squeeze()
        return img.add(rgb.view(3, 1, 1).expand_as(img))

class ImageNet(torch.utils.data.Dataset):
    """ImageNet dataset."""

    def __init__(self, data_path, split, corruption_type, corruption_level):
        assert os.path.exists(data_path), "Data path '{}' not found".format(data_path)
        assert split in [
            "train",
            "val",
        ], "Split '{}' not supported for ImageNet".format(split)
        logger.info("Constructing ImageNet {}...".format(split))

        self._data_path = data_path
        self._split = split
        self._corruption_type = corruption_type
        self._corruption_level = corruption_level
        self._construct_imdb()
        self.img_fmt = cfg.AUG.IMG_FMT

        # self.aug_type = cfg.AUG.TYPE # default to use CHW BGR version
        assert cfg.AUG.TYPE != 'default'
        if self._split == "train":
            self.img_aug = get_aug(cfg.AUG.TYPE)
        else:
            self.img_aug = ttransforms.Compose([
                            ttransforms.Resize(cfg.TEST.IM_SIZE),
                            ttransforms.CenterCrop(cfg.TRAIN.IM_SIZE),
                            ttransforms.ToTensor(),
                            ttransforms.Normalize(_MEAN_RGB, _SD_RGB)])

    def _construct_imdb(self):
        """Constructs the imdb."""
        # Compile the split data path
        if self._corruption_type == 'original':
            split_path = os.path.join(self._data_path, self._split)
            re_pattern = r"^n[0-9]+$"
        else:
            assert self._split == 'val'
            assert self._corruption_level in [1, 2, 3, 4, 5]
            assert self._corruption_type in [
                        'gaussian_noise', 'shot_noise', 'impulse_noise', 
                        'defocus_blur', 'glass_blur', 'motion_blur', 
                        'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 
                        'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
            split_path = os.path.join(self._data_path.replace('imagenet', 'imagenet_c'), self._corruption_type, str(self._corruption_level))
            re_pattern = r"^n[0-9]+$"

        logger.info("{} data path: {}".format(self._split, split_path))
        # Images are stored per class in subdirs (format: n<number>)
        self._class_ids = sorted(f for f in os.listdir(split_path) if re.match(re_pattern, f))
        # Map ImageNet class ids to contiguous ids
        self._class_id_cont_id = {v: i for i, v in enumerate(self._class_ids)}
        # Construct the image db
        self._imdb = []
        for class_id in self._class_ids:
            cont_id = self._class_id_cont_id[class_id]
            im_dir = os.path.join(split_path, class_id)
            for im_name in os.listdir(im_dir):
                self._imdb.append(
                    {"im_path": os.path.join(im_dir, im_name), "class": cont_id}
                )
        logger.info("Number of images: {}".format(len(self._imdb)))
        logger.info("Number of classes: {}".format(len(self._class_ids)))

    def __getitem__(self, index):
        im = Image.open(self._imdb[index]["im_path"])
        im = self.img_aug(im.convert('RGB'))
        if self.img_fmt == 'BGR': # C x H x W
            im = im[[2,1,0], :, :]
        # Retrieve the label
        label = self._imdb[index]["class"]
        return im, label

    def __len__(self):
        return len(self._imdb)