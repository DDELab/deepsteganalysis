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

    def __init__(self, args) -> None:

        super().__init__()
        self.args = args

    def prepare_data(self) -> None:
        # do smth if you need
        pass

    def setup(self, stage: Optional[str] = None):
        args = self.args
        qfs = ['75']
                
        classes = [ ['QF'+str(q)+'/COVER', 'QF'+str(q)+'/JUNI_0.4_bpnzac'  ,'QF'+str(q)+'/UED_0.3_bpnzac'  ] for q in qfs ]
                   
        IL_train = os.listdir(args.dataset.data_path+'QF75/COVER/TRN/')
        IL_val = os.listdir(args.dataset.data_path+'QF75/COVER/VAL/')

        dataset = []
        if args.dataset.pair_constraint:
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
            data_path=args.dataset.data_path,
            kinds=dataset[dataset['fold'] != 0].kind.values,
            image_names=dataset[dataset['fold'] != 0].image_name.values,
            labels=dataset[dataset['fold'] != 0].label.values,
            transforms=get_train_transforms(),
            num_classes=len(classes[0]),
            decoder=args.dataset.decoder
        )
        
        self.valid_dataset = retriever(
            data_path=args.dataset.data_path,
            kinds=dataset[dataset['fold'] == 0].kind.values,
            image_names=dataset[dataset['fold'] == 0].image_name.values,
            labels=dataset[dataset['fold'] == 0].label.values,
            transforms=get_valid_transforms(),
            num_classes=len(classes[0]),
            decoder=args.dataset.decoder
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