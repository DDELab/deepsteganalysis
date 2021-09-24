import os
import sys
import random
import pandas as pd
from typing import Optional, Generator, Union, IO, Dict, Callable
from pathlib import Path
from braceexpand import braceexpand
import argparse
import glob

import pytorch_lightning as pl
from dataloading.retriever import TrainRetriever, TrainRetrieverPaired, get_train_transforms, get_valid_transforms
from torch.utils.data import DataLoader
import torch

class LitStegoDataModule(pl.LightningDataModule):

    def __init__(self, args) -> None:

        super().__init__()
        self.args = args

    def prepare_data(self) -> None:
        # do smth if you need
        pass

    def setup(self, stage: Optional[str] = None):

        dataset = []

        for class_key in self.args.dataset.desc.keys():
            class_datapathes = braceexpand(self.args.dataset.desc[class_key].path)

            for class_datapath in class_datapathes:
                # Add training samples
                full_temp_path = os.path.join(self.args.dataset.data_path,
                                              class_datapath,
                                              self.args.dataset.train_id,
                                              "*" + self.args.dataset.file_ext)
                filelist = glob.glob(full_temp_path)
                
                for a_file in filelist:
                    _, filename = os.path.split(a_file)
                    dataset.append({
                        'kind': os.path.join(class_datapath, self.args.dataset.train_id),
                        'image_name': filename,
                        'label': self.args.dataset.desc[class_key].label,
                        'fold': 1,
                    })

                # Add validation samples
                full_temp_path = os.path.join(self.args.dataset.data_path,
                                              class_datapath,
                                              self.args.dataset.val_id,
                                              "*" + self.args.dataset.file_ext)
                filelist = glob.glob(full_temp_path)
                for a_file in filelist:
                    _, filename = os.path.split(a_file)
                    dataset.append({
                        'kind': os.path.join(class_datapath, self.args.dataset.val_id),
                        'image_name': filename,
                        'label': self.args.dataset.desc[class_key].label,
                        'fold': 0,
                    })

        dataset = pd.DataFrame(dataset)
        
        # register num_classes for later use
        self.num_classes = len(set(dataset.label))
        if self.args.dataset.pair_constraint:
            # group by name 
            dataset = dataset.groupby('image_name').agg(lambda x: x.tolist()).reset_index()
            # make sure fold is not a list
            dataset.fold = dataset.fold.apply(lambda x: x[0])
            retriever = TrainRetrieverPaired
        else:
            retriever = TrainRetriever
        
        # Shuffle
        dataset = dataset.sample(frac=1).reset_index(drop=True)
  
        self.train_dataset = retriever(
            data_path=self.args.dataset.data_path,
            kinds=dataset[dataset['fold'] != 0].kind.values,
            image_names=dataset[dataset['fold'] != 0].image_name.values,
            labels=dataset[dataset['fold'] != 0].label.values,
            transforms=get_train_transforms(),
            num_classes=self.num_classes,
            decoder=self.args.dataset.decoder
        )
        
        self.valid_dataset = retriever(
            data_path=self.args.dataset.data_path,
            kinds=dataset[dataset['fold'] == 0].kind.values,
            image_names=dataset[dataset['fold'] == 0].image_name.values,
            labels=dataset[dataset['fold'] == 0].label.values,
            transforms=get_valid_transforms(),
            num_classes=self.num_classes,
            decoder=self.args.dataset.decoder
        )

    def train_dataloader(self):
        args = self.args
        _dataset = self.train_dataset
        
        def collate_fn(data):
            images, labels = zip(*data)
            images = torch.cat(images)
            labels = torch.cat(labels)
            return images, labels
        
        loader = DataLoader(dataset=_dataset,
                            drop_last=True,
                            batch_size=args.training.batch_size,
                            num_workers=args.dataset.num_workers,
                            collate_fn=collate_fn if args.dataset.pair_constraint else None,
                            shuffle=True)
        return loader

    def val_dataloader(self):
        args = self.args
        _dataset = self.valid_dataset
        
        def collate_fn(data):
            images, labels = zip(*data)
            images = torch.cat(images)
            labels = torch.cat(labels)
            return images, labels
        
        loader = DataLoader(dataset=_dataset,
                            batch_size=args.training.batch_size,
                            num_workers=args.dataset.num_workers,
                            collate_fn=collate_fn if args.dataset.pair_constraint else None,
                            shuffle=False)
        return loader