import os
from typing import Optional, Generator, Union, IO, Dict, Callable
from braceexpand import braceexpand
import glob
import pandas as pd
import torch
from torch.utils.data import DataLoader
from lightning import LightningDataModule

from dataloading.retrievercsv import TrainCSVRetriever
from dataloading.decoders import decoder2in_chans
from dataloading.augmentations import get_train_transforms, get_valid_transforms

def cat_collate_fn(data):
    data = zip(*data)
    return tuple(torch.cat(x) for x in data)

class LitCSVStegoDataModule(LightningDataModule):

    def __init__(self, args) -> None:

        super().__init__()
        self.args = args
        # register in_chans
        self.in_chans = decoder2in_chans(self.args.dataset.decoder)
        # register num_classes for later use
        self.num_classes = self.args.dataset.classes

    def setup(self, stage: Optional[str] = None):

        retriever = TrainCSVRetriever
        self.dataset = pd.read_csv(self.args.dataset.desc)
        self.dataset = self.dataset.sample(frac=1).reset_index(drop=True)
  
        train_id = self.args.dataset.train_id
        self.train_dataset = retriever(
            data_path=self.args.dataset.data_path,
            dataframe=self.dataset.loc[self.dataset.subset == train_id].reset_index(drop=True),
            transforms=get_train_transforms(self.args.dataset.augs_type),
            decoder=self.args.dataset.decoder
        )
        
        val_id = self.args.dataset.val_id
        self.valid_dataset = retriever(
            data_path=self.args.dataset.data_path,
            dataframe=self.dataset.loc[self.dataset.subset == val_id].reset_index(drop=True),
            transforms=get_valid_transforms(self.args.dataset.augs_type),
            decoder=self.args.dataset.decoder
        )
        
        test_id = self.args.dataset.test_id
        self.test_dataset = retriever(
            data_path=self.args.dataset.data_path,
            dataframe=self.dataset.loc[self.dataset.subset == test_id].reset_index(drop=True),
            transforms=get_valid_transforms(self.args.dataset.augs_type),
            return_name=True,
            decoder=self.args.dataset.decoder
        )

    def train_dataloader(self):
        loader = DataLoader(dataset=self.train_dataset,
                            drop_last=True,
                            batch_size=self.args.training.batch_size,
                            num_workers=self.args.dataset.num_workers,
                            shuffle=True)
        return loader

    def val_dataloader(self):
        loader = DataLoader(dataset=self.valid_dataset,
                            batch_size=self.args.training.batch_size,
                            num_workers=self.args.dataset.num_workers,
                            shuffle=False)
        return loader

    def test_dataloader(self):        
        loader = DataLoader(dataset=self.test_dataset,
                            batch_size=self.args.training.batch_size,
                            num_workers=self.args.dataset.num_workers,
                            shuffle=False)
        return loader