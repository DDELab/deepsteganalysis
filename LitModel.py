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
                 backbone: str = 'mixnet_s',
                 lr: float = 1e-3,
                 eps: float = 1e-8,
                 lr_scheduler_name: str = 'cos',
                 surgery: str = '',
                 decay_not_bias_norm: int = 0,
                 optimizer_name: str = 'adamw',
                 epochs: int = 50, 
                 gpus: list = [0], 
                 seed: str = None,
                 weight_decay: float = 1e-2
                 ,**kwargs) -> None:
        
        super().__init__()
        self.epochs = epochs
        self.backbone = backbone
        self.lr = lr
        self.lr_scheduler_name = lr_scheduler_name
        self.optimizer_name = optimizer_name
        self.gpus = len(gpus)
        self.weight_decay = weight_decay
        self.eps = eps
        self.surgery = surgery
        self.seed = seed

        self.decay_not_bias_norm = decay_not_bias_norm
        
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
        #print(acc)       
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
        
        train_len = len(self.trainer.datamodule.train_dataset)
        #print("######### Training len", train_len)
        batch_size = self.trainer.datamodule.batch_size
        #print("########## Batch size", batch_size)

        if self.lr_scheduler_name == 'cos':
            scheduler_kwargs = {'T_max': self.epochs*train_len//self.gpus//batch_size,
                                'eta_min':self.lr/50}

        elif self.lr_scheduler_name == 'onecycle':
            scheduler_kwargs = {'max_lr': self.lr, 'epochs': self.epochs,
                                'steps_per_epoch':train_len//self.gpus//batch_size,
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

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--backbone',
                            default='mixnet_s',
                            type=str,
                            metavar='BK',
                            help='Name (as in ``torchvision.models``) of the feature extractor')
        parser.add_argument('--epochs',
                            default=100,
                            type=int,
                            metavar='N',
                            help='total number of epochs')
        parser.add_argument('--gpus',
                            nargs='+',
                            type=int,
                            default=[0],
                            help='gpus to use')
        parser.add_argument('--decay-not-bias-norm',
                            type=int,
                            default=0,
                            help='do not decay batch norm and bias and FC')
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
