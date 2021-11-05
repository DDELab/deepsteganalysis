import os
import sys
import random
import pandas as pd
import numpy as np
from typing import Optional, Generator, Union, IO, Dict, Callable
from pathlib import Path
from braceexpand import braceexpand
import argparse
import glob

import pytorch_lightning as pl
from dataloading.retriever import TrainRetriever, TrainRetrieverPaired
from dataloading.decoders import decoder2in_chans
from dataloading.augmentations import get_train_transforms, get_valid_transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from dataloading.samplers import DistributedProxySampler
import torch

def cat_collate_fn(data):
    data = zip(*data)
    return tuple(torch.cat(x) for x in data)

class LitStegoDataModule(pl.LightningDataModule):

    def __init__(self, args) -> None:

        super().__init__()
        self.args = args
        # register in_chans
        self.in_chans = decoder2in_chans(self.args.dataset.decoder)
        # register num_classes for later use
        self.num_classes = len(set([self.args.dataset.desc[k].label for k in self.args.dataset.desc.keys()]))

    def prepare_data(self) -> None:
        # do smth if you need
        pass

    def __add_samples(self, class_key, class_datapath, fold_id, fold):
        full_temp_path = os.path.join(self.args.dataset.data_path,
                                      class_datapath,
                                      fold_id,
                                      "*" + self.args.dataset.desc[class_key].file_ext)
        filelist = glob.glob(full_temp_path)

        for a_file in filelist:
            _, filename = os.path.split(a_file)
            self.dataset.append({
                'kind': os.path.join(class_datapath, fold_id),
                'image_name_noext': fold_id+filename.rsplit('.')[0],
                'image_name': filename,
                'label': self.args.dataset.desc[class_key].label,
                'fold': fold,
                'file_type': self.args.dataset.desc[class_key].file_type,
                'payload': self.args.dataset.desc[class_key].payload,
                'file_ext': self.args.dataset.desc[class_key].file_ext,
                'cover_image_name': None,
                'cover_kind': None
            })

    def setup(self, stage: Optional[str] = None):

        self.dataset = []

        for class_key in self.args.dataset.desc.keys():
            class_datapathes = braceexpand(self.args.dataset.desc[class_key].path)

            for class_datapath in class_datapathes:
                # Add training samples
                self.__add_samples(class_key, class_datapath, self.args.dataset.train_id, 1)
                # Add validation samples
                self.__add_samples(class_key, class_datapath, self.args.dataset.val_id, 0)
                # Add test samples
                self.__add_samples(class_key, class_datapath, self.args.dataset.test_id, -1)

        self.dataset = pd.DataFrame(self.dataset)
        
        if self.args.dataset.pair_constraint:
            # group by name 
            self.dataset = self.dataset.groupby('image_name_noext').agg(lambda x: x.tolist()).reset_index()
            # make sure fold is not a list
            self.dataset.fold = self.dataset.fold.apply(lambda x: x[0])
            retriever = TrainRetrieverPaired
        else:
            retriever = TrainRetriever
            # Link proto stego imags (costs, betas) to cover images
            # This relies on entering them in the same order in the dataset desc
            # This also relies on having the number of proto stegos be a multiple of number of covers
            # (i.e.) 1 cover per cost or beta
            len_proto = len(self.dataset.loc[self.dataset.file_type != 'image'])
            len_covers = len(self.dataset[self.dataset.label==0])
            assert len_proto % len_covers == 0
            self.dataset.loc[self.dataset.file_type != 'image', 'cover_image_name'] = self.dataset[self.dataset.label==0].image_name.values.tolist() * (len_proto//len_covers)
            self.dataset.loc[self.dataset.file_type != 'image', 'cover_kind'] = self.dataset[self.dataset.label==0].kind.values.tolist() * (len_proto//len_covers)

        # Shuffle
        self.dataset = self.dataset.sample(frac=1).reset_index(drop=True)

        # make sampler for balanced training
        if self.args.training.balanced:
            assert self.args.dataset.pair_constraint == False, 'pair constraint is always balanced, set it to False or null'
            count = self.dataset.loc[self.dataset.fold == 1].groupby(['label']).count()['image_name'].to_dict()
            balance_weight = np.array([1./count[k] for k in self.dataset.loc[self.dataset.fold == 1, 'label']])
            balance_weight = torch.from_numpy(balance_weight)
            self.sampler = WeightedRandomSampler(balance_weight.type('torch.DoubleTensor'), len(balance_weight))
            if len(self.args.training.gpus or '') > 1:
                self.sampler = DistributedProxySampler(self.sampler)
  
        self.train_dataset = retriever(
            data_path=self.args.dataset.data_path,
            kinds=self.dataset[self.dataset['fold'] == 1].kind.values,
            image_names=self.dataset[self.dataset['fold'] == 1].image_name.values,
            labels=self.dataset[self.dataset['fold'] == 1].label.values,
            file_types=self.dataset[self.dataset['fold'] == 1].file_type.values,
            file_exts=self.dataset[self.dataset['fold'] == 1].file_ext.values,
            payloads=self.dataset[self.dataset['fold'] == 1].payload.values,
            cover_kinds=self.dataset[self.dataset['fold'] == 1].cover_kind.values,
            cover_image_names=self.dataset[self.dataset['fold'] == 1].cover_image_name.values,
            transforms=get_train_transforms(self.args.dataset.augs_type),
            num_classes=self.num_classes,
            decoder=self.args.dataset.decoder
        )
        
        self.valid_dataset = retriever(
            data_path=self.args.dataset.data_path,
            kinds=self.dataset[self.dataset['fold'] == 0].kind.values,
            image_names=self.dataset[self.dataset['fold'] == 0].image_name.values,
            labels=self.dataset[self.dataset['fold'] == 0].label.values,
            file_types=self.dataset[self.dataset['fold'] == 0].file_type.values,
            file_exts=self.dataset[self.dataset['fold'] == 0].file_ext.values,
            payloads=self.dataset[self.dataset['fold'] == 0].payload.values,
            cover_kinds=self.dataset[self.dataset['fold'] == 0].cover_kind.values,
            cover_image_names=self.dataset[self.dataset['fold'] == 0].cover_image_name.values,
            transforms=get_valid_transforms(self.args.dataset.augs_type),
            num_classes=self.num_classes,
            decoder=self.args.dataset.decoder
        )
        
        self.test_dataset = retriever(
            data_path=self.args.dataset.data_path,
            kinds=self.dataset[self.dataset['fold'] == -1].kind.values,
            image_names=self.dataset[self.dataset['fold'] == -1].image_name.values,
            labels=self.dataset[self.dataset['fold'] == -1].label.values,
            file_types=self.dataset[self.dataset['fold'] == -1].file_type.values,
            file_exts=self.dataset[self.dataset['fold'] == -1].file_ext.values,
            payloads=self.dataset[self.dataset['fold'] == -1].payload.values,
            cover_kinds=self.dataset[self.dataset['fold'] == -1].cover_kind.values,
            cover_image_names=self.dataset[self.dataset['fold'] == -1].cover_image_name.values,
            transforms=get_valid_transforms(self.args.dataset.augs_type),
            num_classes=self.num_classes,
            return_name=True,
            decoder=self.args.dataset.decoder
        )

    def train_dataloader(self):
        loader = DataLoader(dataset=self.train_dataset,
                            drop_last=True,
                            batch_size=self.args.training.batch_size,
                            num_workers=self.args.dataset.num_workers,
                            sampler=self.sampler if self.args.training.balanced else None,
                            collate_fn=cat_collate_fn if self.args.dataset.pair_constraint else None,
                            shuffle=True)
        return loader

    def val_dataloader(self):
        loader = DataLoader(dataset=self.valid_dataset,
                            batch_size=self.args.training.batch_size,
                            num_workers=self.args.dataset.num_workers,
                            collate_fn=cat_collate_fn if self.args.dataset.pair_constraint else None,
                            shuffle=False)
        return loader

    def test_dataloader(self):        
        loader = DataLoader(dataset=self.test_dataset,
                            batch_size=self.args.training.batch_size,
                            num_workers=self.args.dataset.num_workers,
                            collate_fn=cat_collate_fn if self.args.dataset.pair_constraint else None,
                            shuffle=False)
        return loader