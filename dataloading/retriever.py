import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import jpegio as jio
import pandas as pd
import numpy as np
import pickle
import cv2
import albumentations as A
import os
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Dataset, DataLoader
import torch
import sys
sys.path.insert(1,'./')
from tools.jpeg_utils import *

DATA_ROOT_PATH = os.environ.get('DATA_ROOT_PATH')

def get_train_transforms():
    return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=1.0),
            #A.Resize(height=256, width=256, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0, additional_targets={'image2':'image'})

def get_valid_transforms():
    return A.Compose([
            #A.Resize(height=256, width=256, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0, additional_targets={'image2':'image'})

def onehot(size, target):
    vec = torch.zeros(size, dtype=torch.float32)
    vec[target] = 1.
    return vec

def decoder2in_chans(decoder):
    if decoder == 'ycbcr':
        return 3
    elif decoder == 'rgb':
        return 3
    elif decoder == 'y':
        return 1

def ycbcr_decode(path):
    tmp = jio.read(path)
    image = decompress_structure(tmp)
    image = image[:,:,:].astype(np.float32)
    image /= 255.0
    return image

def rgb_decode(path):
    tmp = jio.read(path)
    image = decompress_structure(tmp)
    image = image[:,:,:].astype(np.float32)
    image = ycbcr2rgb(image).astype(np.float32)
    image /= 255.0
    return image

def y_decode(path):
    tmp = jio.read(path)
    image = decompress_structure(tmp)
    image = image[:,:,:1].astype(np.float32)
    image /= 255.0
    return image

class TrainRetriever(Dataset):

    def __init__(self, data_path, kinds, image_names, labels, decoder='y', transforms=None, return_name=False, num_classes=2):
        super().__init__()
        
        self.data_path = data_path
        self.kinds = kinds
        self.image_names = image_names
        self.return_name = return_name
        self.labels = labels
        self.transforms = transforms
        self.decoder = decoder
        self.num_classes = num_classes

    def __getitem__(self, index: int):
        
        kind, image_name, label = self.kinds[index], self.image_names[index], self.labels[index]
        if  self.decoder == 'ycbcr':
            image = ycbcr_decode(f'{self.data_path}/{kind}/{image_name}')
        elif  self.decoder == 'rgb':
            image = rgb_decode(f'{self.data_path}/{kind}/{image_name}')
        elif  self.decoder == 'y':
            image = y_decode(f'{self.data_path}/{kind}/{image_name}')
        if self.transforms:
            sample = {'image': image}
            sample = self.transforms(**sample)
            image = sample['image']
                    
        if self.return_name:
            return image, label, image_name
        return image, label

    def __len__(self) -> int:
        return self.image_names.shape[0]

    def get_labels(self):
        return list(self.labels)
    
    
class TrainRetrieverPaired(Dataset):

    def __init__(self, data_path, kinds, image_names, labels, decoder='y', transforms=None, return_name=False, num_classes=2):
        super().__init__()
        
        self.data_path = data_path
        self.kinds = kinds
        self.image_names = image_names
        self.labels = labels
        self.transforms = transforms
        self.decoder = decoder
        self.num_classes = num_classes
        self.return_name = return_name

    def __getitem__(self, index: int):
        
        kind, image_name, label = self.kinds[index], self.image_names[index], self.labels[index]
        i = np.random.randint(low=1, high=self.num_classes)

        if  self.decoder == 'ycbcr':
            cover = ycbcr_decode(f'{self.data_path}/{kind[0]}/{image_name}')
            stego = ycbcr_decode(f'{self.data_path}/{kind[i]}/{image_name}')
        elif  self.decoder == 'rgb':
            cover = ycbcr_decode(f'{self.data_path}/{kind[0]}/{image_name}')
            stego = ycbcr_decode(f'{self.data_path}/{kind[i]}/{image_name}')
        elif  self.decoder == 'y':
            cover = ycbcr_decode(f'{self.data_path}/{kind[0]}/{image_name}')
            stego = ycbcr_decode(f'{self.data_path}/{kind[i]}/{image_name}')

        target_cover = label[0]
        target_stego = label[i]
            
        if self.transforms:
            sample = {'image': cover, 'image2': stego}
            sample = self.transforms(**sample)
            cover = sample['image']
            stego = sample['image2']

        if self.return_name:
            return  torch.stack([cover,stego]), torch.as_tensor([target_cover, target_stego]), torch.as_tensor([image_name, image_name])
        return torch.stack([cover,stego]), torch.as_tensor([target_cover, target_stego])

    def __len__(self) -> int:
        return self.image_names.shape[0]

    def get_labels(self):
        return list(self.labels)