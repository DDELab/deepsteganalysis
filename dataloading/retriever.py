import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import jpegio as jio
import pandas as pd
import numpy as np
import pickle
import random
import cv2
from itertools import compress
import os
from torch.utils.data import Dataset, DataLoader
import torch
import sys
from tools.jpeg_utils import *
from tools.stego_utils import *

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
    elif decoder == 'rjca':
        return 1
    elif decoder == 'gray_spatial':
        return 1
    elif decoder == 'onehot':
        return 6

def load_or_pass(x, type=jio.decompressedjpeg.DecompressedJpeg, load_fn=jio.read):
    return x if isinstance(x, type) else load_fn(x)

def ycbcr_decode(path):
    tmp = load_or_pass(path)
    image = decompress_structure(tmp)
    image = image[:,:,:].astype(np.float32)
    image /= 255.0
    return image

def rgb_decode(path):
    tmp = load_or_pass(path)
    image = decompress_structure(tmp)
    image = image[:,:,:].astype(np.float32)
    image = ycbcr2rgb(image).astype(np.float32)
    image /= 255.0
    return image

def y_decode(path):
    tmp = load_or_pass(path)
    image = decompress_structure(tmp)
    image = image[:,:,:1].astype(np.float32)
    image /= 255.0
    return image

def onehot_decode(path):
    return load_or_pass(path)

def rjca_decode(path):
    tmp = load_or_pass(path)
    image = decompress_structure(tmp)
    image = image[:,:,:1].astype(np.float32)
    return image - np.round(image)

def gray_spatial_decode(path):
    image = load_or_pass(path, np.ndarray, cv2.imread)
    image = image[:,:,:1].astype(np.float32)
    return image

def cost_map_decode(path, cover_path, payload):
    cost_map = np.load(path)
    cover = jio.read(cover_path)
    nzac = np.count_nonzero(cover.coef_arrays[0]) - np.count_nonzero(cover.coef_arrays[0][::8,::8])
    stego = embedding_simulator(cover.coef_arrays[0], cost_map['rho_p1'], cost_map['rho_m1'], nzac*payload)
    cover.coef_arrays[0] = stego
    return cover

def change_map_decode(path, cover_path):
    change_map = np.load(path)
    cover = jio.read(cover_path)
    stego = sample_stego_image(cover.coef_arrays[0], change_map['pChangeP1'], change_map['pChangeM1'])
    cover.coef_arrays[0] = stego
    return cover

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

    def __getitem__(self, index: int):
        
        kind, image_name, label, payload, file_type, cover_kind, cover_image_name = self.kinds[index], self.image_names[index], self.labels[index], self.payloads[index], self.file_types[index], self.cover_kinds[index], self.cover_image_names[index]
        file = f'{self.data_path}/{kind}/{image_name}'
        if file_type == 'cost_map':
            cover_path = f'{self.data_path}/{cover_kind}/{cover_image_name}'
            file = cost_map_decode(file, cover_path, payload)
        if file_type == 'change_map':
            cover_path = f'{self.data_path}/{cover_kind}/{cover_image_name}'
            file = change_map_decode(file, cover_path)
        if  self.decoder == 'ycbcr':
            image = ycbcr_decode(file)
        elif  self.decoder == 'rgb':
            image = rgb_decode(file)
        elif  self.decoder == 'y':
            image = y_decode(file)
        elif  self.decoder == 'onehot':
            image = onehot_decode(file)
        elif  self.decoder == 'gray_spatial':
            image = gray_spatial_decode(file)
        elif  self.decoder == 'rjca':
            image = rjca_decode(file)
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

    def __getitem__(self, index: int):
        
        kind, image_name, label, payload, file_type = self.kinds[index], self.image_names[index], self.labels[index], self.payloads[index], self.file_types[index]
        i = np.random.randint(low=1, high=self.num_classes)
        cover_file = f'{self.data_path}/{kind[0]}/{image_name[0]}'
        stego_file = f'{self.data_path}/{kind[i]}/{image_name[1]}'

        if file_type == 'cost_map':
            stego_file = cost_map_decode(stego_file, cover_file, payload[i])
        if file_type == 'change_map':
            stego_file = change_map_decode(stego_file, cover_file)
        if  self.decoder == 'ycbcr':
            cover = ycbcr_decode(cover_file)
            stego = ycbcr_decode(stego_file)
        elif  self.decoder == 'rgb':
            cover = rgb_decode(cover_file)
            stego = rgb_decode(stego_file)
        elif  self.decoder == 'y':
            cover = y_decode(cover_file)
            stego = y_decode(stego_file)
        elif  self.decoder == 'onehot':
            cover = onehot_decode(cover_file)
            stego = onehot_decode(stego_file)
        elif  self.decoder == 'gray_spatial':
            cover = gray_spatial_decode(cover_file)
            stego = gray_spatial_decode(stego_file)           
        elif  self.decoder == 'rjca':
            cover = rjca_decode(cover_file)
            stego = rjca_decode(stego_file)

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