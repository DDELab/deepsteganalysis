import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import jpegio as jio
import pandas as pd
import numpy as np
import pickle
import random
import cv2
import binascii
from itertools import compress
import os
import dataloading.decoders
from torch.utils.data import Dataset, DataLoader
import torch
import sys
from tools.jpeg_utils import *
from tools.stego_utils import *

def onehot(size, target):
    vec = torch.zeros(size, dtype=torch.float32)
    vec[target] = 1.
    return vec

class TrainRetriever(Dataset):

    def __init__(self, data_path, kinds, image_names, labels, file_types, file_exts, payloads,
                 cover_kinds, cover_image_names, decoder='y', transforms=None, return_name=False, 
                 num_classes=2):
        super().__init__()
        
        self.data_path = data_path
        self.kinds = kinds
        self.image_names = image_names
        self.return_name = return_name
        self.labels = labels
        self.transforms = transforms
        self.decoder = decoder
        self.num_classes = num_classes
        self.file_types = file_types
        self.payloads = payloads
        self.file_exts = file_exts
        self.cover_kinds = cover_kinds
        self.cover_image_names = cover_image_names
        self.decode = getattr(dataloading.decoders, f'{decoder}_decode')    # get fn ptr for decoder

    def __getitem__(self, index: int):
        
        kind, image_name, label, payload, file_type, cover_kind, cover_image_name = self.kinds[index], self.image_names[index], self.labels[index], self.payloads[index], self.file_types[index], self.cover_kinds[index], self.cover_image_names[index]
        file = f'{self.data_path}/{kind}/{image_name}'
        if file_type == 'cost_map':
            cover_path = f'{self.data_path}/{cover_kind}/{cover_image_name}'
            file = dataloading.decoders.cost_map_decode(file, cover_path, payload)
        if file_type == 'change_map':
            cover_path = f'{self.data_path}/{cover_kind}/{cover_image_name}'
            file = dataloading.decoders.change_map_decode(file, cover_path)

        image = self.decode(file)

        if self.transforms:
            sample = {'image': image}
            sample = self.transforms(**sample)
            image = sample['image']
                    
        if self.return_name:
            return image, label, torch.as_tensor(dataloading.decoders.encode_string(image_name))
        return image, label

    def __len__(self) -> int:
        return self.image_names.shape[0]

    def get_labels(self):
        return list(self.labels)
    
    
class TrainRetrieverPaired(Dataset):

    def __init__(self, data_path, kinds, image_names, labels, file_types, file_exts, payloads,
                 cover_kinds, cover_image_names, decoder='y', transforms=None, return_name=False,
                 num_classes=2):
        super().__init__()
        
        self.data_path = data_path
        self.kinds = kinds
        self.image_names = image_names
        self.labels = labels
        self.transforms = transforms
        self.decoder = decoder
        self.num_classes = num_classes
        self.return_name = return_name
        self.payloads = payloads
        self.file_types = file_types
        self.file_exts = file_exts
        self.decode = getattr(dataloading.decoders, f'{decoder}_decode')    # get fn ptr for decoder

    def __getitem__(self, index: int):
        
        kind, image_name, label, payload, file_type = self.kinds[index], self.image_names[index], self.labels[index], self.payloads[index], self.file_types[index]
        i = np.random.randint(low=1, high=self.num_classes)
        cover_file = f'{self.data_path}/{kind[0]}/{image_name[0]}'
        stego_file = f'{self.data_path}/{kind[i]}/{image_name[i]}'

        if file_type == 'cost_map':
            stego_file = dataloading.decoders.cost_map_decode(stego_file, cover_file, payload[i])
        if file_type == 'change_map':
            stego_file = dataloading.decoders.change_map_decode(stego_file, cover_file)
        
        cover = self.decode(cover_file)
        stego = self.decode(stego_file)

        target_cover = label[0]
        target_stego = label[i]
            
        if self.transforms:
            sample = {'image': cover, 'image2': stego}
            sample = self.transforms(**sample)
            cover = sample['image']
            stego = sample['image2']

        if self.return_name:
            return  torch.stack([cover,stego]), \
                    torch.as_tensor([target_cover, target_stego]), \
                    torch.as_tensor([dataloading.decoders.encode_string(image_name[0]), dataloading.decoders.encode_string(image_name[i])])
        return torch.stack([cover,stego]), torch.as_tensor([target_cover, target_stego])

    def __len__(self) -> int:
        return self.image_names.shape[0]

    def get_labels(self):
        return list(self.labels)