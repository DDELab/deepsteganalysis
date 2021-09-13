import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np 
import pickle
import argparse
from collections import OrderedDict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, Generator, Union, IO, Dict, Callable
import torch
import torch.nn.functional as F
from torch import optim
from torch.nn import Module
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import _logger as log
import random
from optimizers import *
import models
from retriever import *
from custom_metrics import RocAucMeter, PEMeter, MD5Meter
from pytorch_lightning.utilities.cloud_io import load as pl_load
from pytorch_lightning.utilities.distributed import sync_ddp_if_available
    
class LitModel(pl.LightningModule):
    """Transfer Learning
    """
    def __init__(self,
                 data_path: Union[str, Path],
                 backbone: str = 'mixnet_s',
                 batch_size: int = 32,
                 lr: float = 1e-3,
                 eps: float = 1e-8,
                 lr_scheduler_name: str = 'cos',
                 surgery: str = '',
                 decay_not_bias_norm: int = 0,
                 pair_constraint: int = 0,
                 qf: str = '',
                 all_qfs: int = 1, 
                 optimizer_name: str = 'adamw',
                 decoder: str = 'NR',
                 num_workers: int = 6, 
                 epochs: int = 50, 
                 gpus: list = [0], 
                 seed: str = None,
                 weight_decay: float = 1e-2
                 ,**kwargs) -> None:
        
        super().__init__()
        self.data_path = data_path
        self.epochs = epochs
        self.backbone = backbone
        self.batch_size = batch_size
        self.lr = lr
        self.qf = qf
        self.all_qfs = all_qfs
        self.num_workers = num_workers
        self.lr_scheduler_name = lr_scheduler_name
        self.optimizer_name = optimizer_name
        self.gpus = len(gpus)
        self.weight_decay = weight_decay
        self.eps = eps
        self.surgery = surgery
        self.seed = seed
        self.decoder = decoder

        self.decay_not_bias_norm = decay_not_bias_norm
        self.pair_constraint = pair_constraint 
        
        self.save_hyperparameters()
        
        self.train_custom_metrics = {'train_wAUC': RocAucMeter(), 'train_mPE': PEMeter()}
        self.validation_custom_metrics = {'val_wAUC': RocAucMeter(), 'val_mPE': PEMeter(), 'val_MD5': MD5Meter()}
        
        self.__build_model()

    def __build_model(self):
        """Define model layers & loss."""

        # 1. Load pre-trained network:
        self.net = models.get_net(self.backbone)
        
        if self.seed is not None:
            checkpoint = torch.load(self.seed)
            model_dict = self.net.state_dict()
            state_dict = {k.split('net.')[1]: v for k, v in checkpoint['state_dict'].items() if 'classifier.' not in k}
            model_dict.update(state_dict)
            self.net.load_state_dict(model_dict)
            print('loaded seed checkpoint')
            del checkpoint
            del model_dict
            del state_dict
        
        if self.surgery != '':
            self.net = getattr(models, self.surgery)(self.net)

        # 2. Loss:
        self.loss_func = F.cross_entropy

    def forward(self, x):
        """Forward pass. Returns logits."""

        x = self.net(x)
        
        return x

    def loss(self, logits, labels):
        return self.loss_func(logits, torch.argmax(labels, dim=1))

    def training_step(self, batch, batch_idx):

        # 1. Forward pass:
        x, y = batch
        y_logits = self.forward(x)
        
        pred_bin = torch.argmax(y_logits, dim=1)
        pred_bin = pred_bin > 0
        
        y_bin = torch.argmax(y, dim=1)
        y_bin = y_bin > 0
        
        # 2. Compute loss & accuracy:
        train_loss = self.loss(y_logits, y)
        
        # metrics = {'loss': train_loss}
        
        metrics = {}
        for metric_name in self.train_custom_metrics.keys():
            metrics[metric_name] = self.train_custom_metrics[metric_name].update(y_logits, y)
            
        acc = torch.eq(y_bin.view(-1), pred_bin.view(-1)).float().mean()
        metrics['acc'] = acc
        # 3. Outputs: 
        print(acc)       
        self.log("train_loss", train_loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log("acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        output = OrderedDict({'loss': train_loss,
                              'acc': acc,
                              'log': metrics,
                              'progress_bar': metrics})
        return output

    def validation_step(self, batch, batch_idx):
        
        # 1. Forward pass:
        x, y = batch
        y_logits = self.forward(x)
        pred_bin = torch.argmax(y_logits, dim=1)
        pred_bin = pred_bin > 0
        
        y_bin = torch.argmax(y, dim=1)
        y_bin = y_bin > 0
        # 2. Compute loss & accuracy:
        val_loss = self.loss(y_logits, y)
        
        metrics = {'val_loss': val_loss}
        self.log('val_loss', val_loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        for metric_name in self.validation_custom_metrics.keys():
            metrics[metric_name] = self.validation_custom_metrics[metric_name].update(y_logits, y)
            self.log(metric_name, metrics[metric_name], on_step=True, on_epoch=True, logger=True, sync_dist=True)
            
        acc = torch.eq(y_bin.view(-1), pred_bin.view(-1)).float().mean()
        metrics['val_acc'] = acc
        self.log('val_acc', acc, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        
        return metrics

    def test_step(self, batch, batch_idx):
        x, y, name = batch
        y_logits = self.forward(x)
        y_pred = 1 - F.softmax(y_logits.double(), dim=1)[:,0]
        
        result = pl.EvalResult()
        result.write('preds_logit', y_pred, filename='predictions.txt')
        result.write('label', torch.argmax(y, dim=1), filename='predictions.txt')
        result.write('name', list(name), filename='predictions.txt')
        return result
        
    def configure_optimizers(self):
        optimizer = get_optimizer(self.optimizer_name)
        
        optimizer_kwargs = {'momentum': 0.9} if self.optimizer_name == 'sgd' else {'eps': self.eps}
        
        param_optimizer = list(self.net.named_parameters())
        
        if self.decay_not_bias_norm:
            no_decay = ['bias', 'norm.bias', 'norm.weight', 'fc.weight', 'fc.bias']
        else:
            no_decay = []
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': self.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ] 
        optimizer = optimizer(optimizer_grouped_parameters, 
                              lr=self.lr, 
                              **optimizer_kwargs)
        
        if self.lr_scheduler_name == 'cos':
            scheduler_kwargs = {'T_max': self.epochs*len(self.train_dataset)//self.gpus//self.batch_size,
                                'eta_min':self.lr/50}

        elif self.lr_scheduler_name == 'onecycle':
            scheduler_kwargs = {'max_lr': self.lr, 'epochs': self.epochs,
                                'steps_per_epoch':len(self.train_dataset)//self.gpus//self.batch_size,
                                'pct_start':4.0/self.epochs,'div_factor':25,'final_div_factor':2}
                                #'div_factor':25,'final_div_factor':2}

        elif self.lr_scheduler_name == 'multistep':
             scheduler_kwargs = {'milestones':[350]}

        elif self.lr_scheduler_name == 'const':
            scheduler_kwargs = {'lr_lambda': lambda epoch: 1}
            
        scheduler = get_lr_scheduler(self.lr_scheduler_name)
        scheduler_params, interval = get_lr_scheduler_params(self.lr_scheduler_name, **scheduler_kwargs)
        scheduler = scheduler(optimizer, **scheduler_params)

        return [optimizer], [{'scheduler':scheduler, 'interval': interval, 'name': 'lr'}]

    def prepare_data(self):
        """Download images and prepare images datasets."""
        
        print('data downloaded')

    def setup(self, stage: str): 
        
        qfs = ['75']
        
        if not self.all_qfs:
            qfs = [self.qf]
        
        classes = [ ['QF'+str(q)+'/COVER', 'QF'+str(q)+'/JUNI_0.4_bpnzac'  ,'QF'+str(q)+'/UED_0.3_bpnzac'  ] for q in qfs ]
                   
                           
        IL_train = os.listdir(self.data_path+'QF75/COVER/TRN/')[:24]
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
    
    def __dataloader(self, train):
        """Train/validation loaders."""

        _dataset = self.train_dataset if train else self.valid_dataset
        
        def collate_fn(data):
            images, labels = zip(*data)
            images = torch.cat(images)
            labels = torch.cat(labels)
            return images, labels
        
        loader = DataLoader(dataset=_dataset,
                            batch_size=self.batch_size,
                            num_workers=self.num_workers,
                            collate_fn=collate_fn if self.pair_constraint else None,
                            shuffle=True if train else False)

        return loader

    def train_dataloader(self):
        log.info('Training data loaded.')
        return self.__dataloader(train=True)

    def val_dataloader(self):
        log.info('Validation data loaded.')
        return self.__dataloader(train=False)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser])
        parser.add_argument('--backbone',
                            default='mixnet_s',
                            type=str,
                            metavar='BK',
                            help='Name (as in ``torchvision.models``) of the feature extractor')
        parser.add_argument('--decoder',
                            default='NR',
                            type=str)
        parser.add_argument('--data-path',
                            default='/media/multi_quality_factor/JPEG_standard/',
                            type=str,
                            metavar='dp',
                            help='data_path')
        parser.add_argument('--epochs',
                            default=100,
                            type=int,
                            metavar='N',
                            help='total number of epochs')
        parser.add_argument('--batch-size',
                            default=32,
                            type=int,
                            metavar='B',
                            help='batch size',
                            dest='batch_size')
        parser.add_argument('--gpus',
                            nargs='+',
                            type=int,
                            default=[0],
                            help='gpus to use')
        parser.add_argument('--decay-not-bias-norm',
                            type=int,
                            default=0,
                            help='do not decay batch norm and bias and FC')
        parser.add_argument('--pair-constraint',
                            type=int,
                            default=0,
                            help='Use pair constraint?')
        parser.add_argument('--lr',
                            '--learning-rate',
                            default=1e-3,
                            type=float,
                            metavar='LR',
                            help='initial learning rate',
                            dest='lr')
        parser.add_argument('--eps',
                            default=1e-8,
                            type=float,
                            help='eps for adaptive optimizers',
                            dest='eps')
        parser.add_argument('--num-workers',
                            default=6,
                            type=int,
                            metavar='W',
                            help='number of CPU workers',
                            dest='num_workers')
        parser.add_argument('--lr-scheduler-name',
                            default='cos',
                            type=str,
                            metavar='LRS',
                            help='Name of LR scheduler')
        parser.add_argument('--optimizer-name',
                            default='adamw',
                            type=str,
                            metavar='OPTI',
                            help='Name of optimizer')
        parser.add_argument('--surgery',
                            default='',
                            type=str,
                            help='name of surgery function')
        parser.add_argument('--qf',
                            default='',
                            type=str,
                            help='quality factor')
        parser.add_argument('--all-qfs',
                            default=1,
                            type=int,
                            help='train on all QFs?')
        parser.add_argument('--weight-decay',
                            default=1e-2,
                            type=float,
                            metavar='wd',
                            help='Optimizer weight decay')
        parser.add_argument('--seed',
                            default=None,
                            type=str,
                            help='path to seeding checkpoint')

        return parser
