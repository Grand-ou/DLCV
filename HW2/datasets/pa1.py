
import os

import cv2
import numpy as np
from copy import deepcopy
import torch
from torch.utils.data import Dataset
from torchvision import transforms

torch.cuda.empty_cache()


class ImageDataset(Dataset):
    def __init__(self, root_dir, transform):
        self.root_dir = root_dir
        self.transform = transform
        file_array = []
        label_array = []
        for filename in os.listdir(root_dir):

            label = filename.split('_')[0]
            file_array.append(filename)
            label_array.append(int(label))
        self.file_array = file_array
        self.label_array = label_array


    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        # Loading and preprocessing.
        label = np.array(self.label_array[index])
        
        img = self.load_frames(self.file_array[index])
        img = self.transform(img)
        label = torch.from_numpy(label)
        label = label.type(torch.LongTensor)
        return img, label

    def randomflip(self, buffer):
        """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""
        if np.random.random() < 0.5:
            for i, frame in enumerate(buffer):
                frame = cv2.flip(buffer[i], flipCode=1)
                buffer[i] = cv2.flip(frame, flipCode=1)
        return buffer


    def load_frames(self, frame_name):
        frame = cv2.imread(os.path.join(self.root_dir, frame_name) )
        print(frame.shape)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype(np.float64)
        return frame

class Normalize(object):
    def __init__(self, RGB_mean=[0.4914, 0.4822, 0.4465], RGB_std=[0.2023, 0.1994, 0.2010]):
        self.RGB_mean = RGB_mean
        self.RGB_std = RGB_std

    def __call__(self, sample):
        images = deepcopy(sample)
        R_mean, G_mean, B_mean = self.RGB_mean[0], self.RGB_mean[1], self.RGB_mean[2]
        R_std, G_std, B_std = self.RGB_std[0], self.RGB_std[1], self.RGB_std[2]
        images[:,:,0] = (images[:,:,0]/255 - R_mean)/R_std
        images[:,:,1] = (images[:,:,1]/255 - G_mean)/G_std
        images[:,:,2] = (images[:,:,2]/255 - B_mean)/B_std
        return images


class Transpose(object):
    def __call__(self, sample):
        return sample.permute((2, 0, 1))



class ToTensor(object):
    def __call__(self, sample):
        return torch.from_numpy(sample)


if __name__ == "__main__":
    transform = transforms.Compose([    
            ToTensor(),
            Normalize(),
            Transpose()
        ])
    train_data = ImageDataset(root_dir = '/content/drive/MyDrive/DLCV/HW1/hw1_data/p1_data/train_50',transform=transform)
    print(train_data[4][0].shape)
    print(train_data[4])