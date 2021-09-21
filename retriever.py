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
            A.Resize(height=256, width=256, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0, additional_targets={'image2':'image'})

def get_valid_transforms():
    return A.Compose([
            A.Resize(height=256, width=256, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0, additional_targets={'image2':'image'})

def onehot(size, target):
    vec = torch.zeros(size, dtype=torch.float32)
    vec[target] = 1.
    return vec

class TrainRetriever(Dataset):

    def __init__(self, data_path, kinds, image_names, labels, decoder='NR', transforms=None, return_name=False, num_classes=2):
        super().__init__()
        
        self.data_path = data_path
        self.kinds = kinds
        self.image_names = image_names
        self.return_name = return_name
        self.labels = labels
        self.transforms = transforms
        self.decoder = decoder
        self.num_classes = num_classes
        self.mean = np.array([0.3914976, 0.44266784, 0.46043398])
        self.std = np.array([0.17819773, 0.17319807, 0.18128773])

    def __getitem__(self, index: int):
        
        kind, image_name, label = self.kinds[index], self.image_names[index], self.labels[index]
        
        if  self.decoder == 'NR':
            tmp = jio.read(f'{self.data_path}/{kind}/{image_name}')
            image = decompress_structure(tmp)
            image = image[:,:,:1].astype(np.float32)
            #image = ycbcr2rgb(image).astype(np.float32)
            image /= 255.0
        elif self.decoder == 'NRYYY':
            tmp = jio.read(f'{self.data_path}/{kind}/{image_name}')
            image = decompress_structure(tmp)
            image = image[:,:,[0,0,0]].astype(np.float32)
            image /= 255.0
        else:
            image = cv2.imread(f'{self.data_path}/{kind}/{image_name}', cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
            image /= 255.0
        if self.transforms:
            sample = {'image': image}
            sample = self.transforms(**sample)
            image = sample['image']
            
        #label = onehot(self.num_classes, label)
        
        if self.return_name:
            return image, label, image_name
        else:
            return image, label

    def __len__(self) -> int:
        return self.image_names.shape[0]

    def get_labels(self):
        return list(self.labels)
    
    
    
class TrainRetrieverPaired(Dataset):

    def __init__(self, data_path, kinds, image_names, labels, decoder='NR', transforms=None, num_classes=2):
        super().__init__()
        
        self.data_path = data_path
        self.kinds = kinds
        self.image_names = image_names
        self.labels = labels
        self.transforms = transforms
        self.decoder = decoder
        self.num_classes = num_classes

    def __getitem__(self, index: int):
        
        kind, image_name, label = self.kinds[index], self.image_names[index], self.labels[index]
        
        tmp = jio.read(f'{self.data_path}/{kind[0]}/{image_name[0]}')
        cover = decompress_structure(tmp)
        cover = cover[:,:,:1].astype(np.float32)
        #cover = ycbcr2rgb(cover).astype(np.float32)
        cover /= 255.0
        target_cover = label[0]
        
        i = np.random.randint(low=1, high=self.num_classes)
        tmp = jio.read(f'{self.data_path}/{kind[i]}/{image_name[i]}')
        stego = decompress_structure(tmp)
        stego = stego[:,:,:1].astype(np.float32)
        #stego = ycbcr2rgb(stego).astype(np.float32)
        stego /= 255.0
        target_stego = label[i]
            
        if self.transforms:
            sample = {'image': cover, 'image2': stego}
            sample = self.transforms(**sample)
            cover = sample['image']
            stego = sample['image2']
            
        return torch.stack([cover,stego]), [target_cover, target_stego]

    def __len__(self) -> int:
        return self.image_names.shape[0]

    def get_labels(self):
        return list(self.labels)
    
    
class TestRetriever(Dataset):

    def __init__(self, data_path, kinds, image_names, labels, decoder='NR', transforms=None):
        super().__init__()
        self.data_path = data_path
        self.test_data_path = self.data_path+'Test/'
        self.kinds = kinds
        self.image_names = image_names
        self.labels = labels
        self.transforms = transforms
        self.decoder = decoder

    def __getitem__(self, index: int):
        image_name = self.image_names[index]
        
        if  self.decoder == 'NR':
            tmp = jio.read(f'{self.test_data_path}/{image_name}')
            image = decompress_structure(tmp).astype(np.float32)
            image = image[:,:,:1].astype(np.float32)
            #image = ycbcr2rgb(image)
            image /= 255.0
        else:
            image = cv2.imread(f'{self.folder}/{image_name}', cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
            image /= 255.0
            
        image = self.func_transforms(image)

        if self.transforms:
            sample = {'image': image}
            sample = self.transforms(**sample)
            image = sample['image']

        return image_name, image

    def __len__(self) -> int:
        return self.image_names.shape[0]
    