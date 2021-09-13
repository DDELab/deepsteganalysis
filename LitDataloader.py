import os
import random
import pandas as pd
from typing import Optional, Generator, Union, IO, Dict, Callable
from pathlib import Path
import argparse 

import pytorch_lightning as pl
from retriever import TrainRetriever, TrainRetrieverPaired
from retriever import get_train_transforms, get_valid_transforms
from torch.utils.data import DataLoader
import torch

class LitStegoDataModule(pl.LightningDataModule):

    def __init__(self,
                 data_path : Union[str, Path],
                 batch_size: int = 32,
                 decoder='NR',
                 pair_constraint: int = 0,
                 qf: str = '',
                 all_qfs: int = 1, 
                 num_workers: int = 6,
                 **kwargs) -> None:

        super().__init__()
        self.data_path = data_path
        self.decoder = decoder
        self.pair_constraint = pair_constraint
        self.qf = qf
        self.all_qfs = all_qfs
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self) -> None:
        # do smth if you need
        pass

    def setup(self, stage: Optional[str] = None):
        qfs = ['75']
        
        if not self.all_qfs:
            qfs = [self.qf]
        
        classes = [ ['QF'+str(q)+'/COVER', 'QF'+str(q)+'/JUNI_0.4_bpnzac'  ,'QF'+str(q)+'/UED_0.3_bpnzac'  ] for q in qfs ]
                   
        IL_train = os.listdir(self.data_path+'QF75/COVER/TRN/')[:960]
        IL_val = os.listdir(self.data_path+'QF75/COVER/VAL/')

        dataset = []
        if self.pair_constraint:
            retriever = TrainRetrieverPaired
            for cl in classes:
                for path in IL_train:
                    dataset.append({
                        'kind': tuple([c + '/TRN' for c in cl]),
                        'image_name': (path, path, path, path),
                        'label': (0,1,2),
                        'fold':1,
                    })
            for cl in classes:
                for path in IL_val:
                    dataset.append({
                        'kind': tuple([c + '/VAL' for c in cl]),
                        'image_name': (path, path, path, path),
                        'label': (0,1,2),
                        'fold':0,
                    })
        else:
            retriever = TrainRetriever
            for cl in classes:
                for label, kind in enumerate(cl):
                    for path in IL_train:
                        dataset.append({
                            'kind': kind+'/TRN',
                            'image_name': path,
                            'label': label,
                            'fold':1,
                        })
            for cl in classes:
                for label, kind in enumerate(cl):
                    for path in IL_val:
                        dataset.append({
                            'kind': kind+'/VAL',
                            'image_name': path,
                            'label': label,
                            'fold':0,
                        })

        random.shuffle(dataset)
        dataset = pd.DataFrame(dataset)
        
        self.train_dataset = retriever(
            data_path=self.data_path,
            kinds=dataset[dataset['fold'] != 0].kind.values,
            image_names=dataset[dataset['fold'] != 0].image_name.values,
            labels=dataset[dataset['fold'] != 0].label.values,
            transforms=get_train_transforms(),
            num_classes=len(classes[0]),
            decoder=self.decoder
        )
        
        self.valid_dataset = retriever(
            data_path=self.data_path,
            kinds=dataset[dataset['fold'] == 0].kind.values,
            image_names=dataset[dataset['fold'] == 0].image_name.values,
            labels=dataset[dataset['fold'] == 0].label.values,
            transforms=get_valid_transforms(),
            num_classes=len(classes[0]),
            decoder=self.decoder
        )

    def train_dataloader(self):
        _dataset = self.train_dataset
        
        def collate_fn(data):
            images, labels = zip(*data)
            images = torch.cat(images)
            labels = torch.cat(labels)
            return images, labels
        
        loader = DataLoader(dataset=_dataset,
                            batch_size=self.batch_size,
                            num_workers=self.num_workers,
                            collate_fn=collate_fn if self.pair_constraint else None,
                            shuffle=True)
        return loader

    def val_dataloader(self):
        _dataset = self.valid_dataset
        
        def collate_fn(data):
            images, labels = zip(*data)
            images = torch.cat(images)
            labels = torch.cat(labels)
            return images, labels
        
        loader = DataLoader(dataset=_dataset,
                            batch_size=self.batch_size,
                            num_workers=self.num_workers,
                            collate_fn=collate_fn if self.pair_constraint else None,
                            shuffle=False)
        return loader

    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--decoder',
                            default='NR',
                            type=str)
        parser.add_argument('--data-path',
                            default='/media/multi_quality_factor/JPEG_standard/',
                            type=str,
                            metavar='dp',
                            help='data_path')
        parser.add_argument('--batch-size',
                            default=32,
                            type=int,
                            metavar='B',
                            help='batch size',
                            dest='batch_size')
        parser.add_argument('--pair-constraint',
                            type=int,
                            default=0,
                            help='Use pair constraint?')
        parser.add_argument('--qf',
                            default='',
                            type=str,
                            help='quality factor')
        parser.add_argument('--all-qfs',
                            default=1,
                            type=int,
                            help='train on all QFs?')
        parser.add_argument('--num-workers',
                            default=6,
                            type=int,
                            metavar='W',
                            help='number of CPU workers',
                            dest='num_workers')

        return parser
