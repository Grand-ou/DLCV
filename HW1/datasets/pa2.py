"""
Dataset class.

Library:       Tensowflow 2.2.0, pyTorch 1.5.1, OpenCV-Python 4.1.1.26
Author:        Ian Yoo
Email:        thyoostar@gmail.com
"""
from __future__ import absolute_import, print_function, division
import os
import numpy as np
import time
import torch
import imageio
from torch.utils.data import Dataset
import cv2
from util.mean_iou_evaluate import read_masks
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class DataLoaderError(Exception):
    pass

try:
    from tqdm import tqdm
except ImportError:
    print("tqdm not found, disabling progress bars")

    def tqdm(iter):
        return iter

TQDM_COLS = 80

class SegmentationDataset(Dataset):

    def __init__(self, images_dir, n_classes, transform=None, mode='train'):
        
        super(SegmentationDataset, self).__init__()
        self.mode = mode
        self.images_dir = images_dir
        self.transform = transform
        self.n_classes = n_classes
        J = 1

        self.filenames = self._get_image_pairs_(self.images_dir)
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
               idx = idx.tolist()
        filename = self.filenames[idx]
        if self.mode == 'train':
            im = cv2.imread(os.path.join(self.images_dir, filename+'_sat.jpg'), flags=cv2.IMREAD_COLOR)
            mask = imageio.imread(os.path.join(os.path.join(self.images_dir, filename+'_mask.png')))
            mask = (mask >= 128).astype(int)
            FD = 1
            mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]
            masks = np.empty((512, 512))
            masks[mask == 3] = 0  # (Cyan: 011) Urban land 
            masks[mask == 6] = 1  # (Yellow: 110) Agriculture land 
            masks[mask == 5] = 2  # (Purple: 101) Rangeland 
            masks[mask == 2] = 3  # (Green: 010) Forest land 
            masks[mask == 1] = 4  # (Blue: 001) Water 
            masks[mask == 7] = 5  # (White: 111) Barren land 
            masks[mask == 0] = 6  # (Black: 000) Unknown 
            sample = {'image': im, 'labeled': masks[0]}

            if self.transform:
                sample = self.transform(sample)

            return sample
        else:
            im = cv2.imread(os.path.join(self.images_dir, filename+'.jpg'), flags=cv2.IMREAD_COLOR)
            print()
            sample = {'image': im, 'labeled': filename}

            if self.transform:
                sample = self.transform(sample)

            return sample
    def _get_image_pairs_(self, img_path):
        """ Check two images have the same name and get all the images
        :param img_path1: directory
        :param img_path2: directory
        :return: pair paths
        """

        AVAILABLE_IMAGE_FORMATS = [".jpg", ".jpeg", ".png", ".bmp"]

        files = []
        
        for dir_entry in os.listdir(img_path):
            if self.mode == 'train':    
                file_name, file_extension = os.path.splitext(dir_entry)[0].split('_')
                if file_extension == 'sat':
                    files.append(file_name)  
            else:
                files.append(os.path.splitext(dir_entry)[0]) 
 
                
        return files
    