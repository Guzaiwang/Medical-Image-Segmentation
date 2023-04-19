import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
import SimpleITK as sitk
import yaml
import math
import random
import pdb
from training import augmentation
import logging
import copy
import cv2
from PIL import Image
from .dataset_drive_utils import randomRotate90, randomVerticleFlip, randomHorizontalFlip, randomShiftScaleRotate, randomHueSaturationValue


class DRIVEDataset(Dataset):
    def __init__(self, args, mode='train'):

        self.mode = mode
        self.args = args

        assert mode in ['train', 'test']
        img_path_list = []
        msk_path_list = []

        if mode == 'train':
            img_folder = '/DATA/i2r/guzw/dataset/DRIVE/training/images'
            mask_folder = '/DATA/i2r/guzw/dataset/DRIVE/training/masks'

            for img_name in os.listdir(img_folder):
                img_name_path = os.path.join(img_folder, img_name)
                img_path_list.append(img_name_path)
                msk_name_path = os.path.join(mask_folder, img_name.replace('_training.tif', '_manual1.png'))
                msk_path_list.append(msk_name_path)
                assert len(img_path_list) == len(msk_path_list)
            
        else:
            img_folder = '/DATA/i2r/guzw/dataset/DRIVE/test/images'
            mask_folder = '/DATA/i2r/guzw/dataset/DRIVE/test/masks'

            for img_name in os.listdir(img_folder):
                img_name_path = os.path.join(img_folder, img_name)
                img_path_list.append(img_name_path)
                msk_name_path = os.path.join(mask_folder, img_name.replace('_training.tif', '_manual1.png'))
                msk_path_list.append(msk_name_path)
                assert len(img_path_list) == len(msk_path_list)

        self.img_path_list = img_path_list
        self.msk_path_list = msk_path_list

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, idx):

        tensor_img_path = self.img_path_list[idx]
        tensor_lab_path = self.msk_path_list[idx]

        img = cv2.imread(tensor_img_path)
        
        mask = np.array(Image.open(tensor_lab_path))

        if self.mode == 'train':
            
            img = cv2.resize(img, (448, 448))
            mask = cv2.resize(mask, (448, 448))
            img = randomHueSaturationValue(img,
                                    hue_shift_limit=(-30, 30),
                                    sat_shift_limit=(-5, 5),
                                    val_shift_limit=(-15, 15))

            img, mask = randomShiftScaleRotate(img, mask,
                                        shift_limit=(-0.1, 0.1),
                                        scale_limit=(-0.1, 0.1),
                                        aspect_limit=(-0.1, 0.1),
                                        rotate_limit=(-0, 0))
            img, mask = randomHorizontalFlip(img, mask)
            img, mask = randomVerticleFlip(img, mask)
            img, mask = randomRotate90(img, mask)

        mask = np.expand_dims(mask, axis=2)
        img = np.array(img, np.float32).transpose(2, 0, 1) / 255.0 * 3.2 - 1.6
        mask = np.array(mask, np.float32).transpose(2, 0, 1)
        mask[mask >= 0.5] = 1
        mask[mask <= 0.5] = 0

        img = torch.from_numpy(img)
        mask = torch.from_numpy(mask).long()

        return img, mask

