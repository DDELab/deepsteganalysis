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
    def __init__(self, args) -> None:
        
        super().__init__()
        self.args = args  
        self.save_hyperparameters(args)
        
        self.train_custom_metrics = {'train_wAUC': RocAucMeter(), 'train_mPE': PEMeter()}
        self.validation_custom_metrics = {'val_wAUC': RocAucMeter(), 'val_mPE': PEMeter(), 'val_MD5': MD5Meter()}
        
        self.__build_model()

    def __build_model(self):
        """Define model layers & loss."""
        args = self.args

        # 1. Load pre-trained network:
        self.net = models.get_net(args.model.backbone)

        if args.ckpt.seed_from is not None and args.ckpt.seed_from != "":
            checkpoint = torch.load(args.ckpt.seed_from)
            model_dict = self.net.state_dict()
            state_dict = {k.split('net.')[1]: v for k, v in checkpoint['state_dict'].items() if 'classifier.' not in k}
            model_dict.update(state_dict)
            self.net.load_state_dict(model_dict)
            print('loaded seed checkpoint')
            del checkpoint
            del model_dict
            del state_dict
        
        if args.model.surgery != '':
            self.net = getattr(models, args.model.surgery)(self.net)

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
        args = self.args

        optimizer = get_optimizer(args.optimizer.name)
        
        optimizer_kwargs = {'momentum': 0.9} if args.optimizer.name == 'sgd' else {'eps': args.optimizer.eps}
        
        param_optimizer = list(self.net.named_parameters())
        
        if args.optimizer.decay_not_bias_norm:
            no_decay = ['bias', 'norm.bias', 'norm.weight', 'fc.weight', 'fc.bias']
        else:
            no_decay = []
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.optimizer.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ] 
        optimizer = optimizer(optimizer_grouped_parameters, 
                              lr=args.optimizer.lr, 
                              **optimizer_kwargs)
        
        train_len = len(self.trainer.datamodule.train_dataset)
        #print("######### Training len", train_len)
        batch_size = args.training.batch_size
        #print("########## Batch size", batch_size)

        if args.optimizer.lr_scheduler_name == 'cos':
            scheduler_kwargs = {'T_max': args.training.epochs*train_len//len(args.training.gpus)//batch_size,
                                'eta_min':args.optimizer.lr/50}

        elif args.optimizer.lr_scheduler_name == 'onecycle':
            scheduler_kwargs = {'max_lr': args.optimizer.lr, 'epochs': args.training.epochs,
                                'steps_per_epoch':train_len//len(args.training.gpus)//batch_size,
                                'pct_start':4.0/args.training.epochs,'div_factor':25,'final_div_factor':2}
                                #'div_factor':25,'final_div_factor':2}

        elif args.optimizer.lr_scheduler_name == 'multistep':
             scheduler_kwargs = {'milestones':[350]}

        elif args.optimizer.lr_scheduler_name == 'const':
            scheduler_kwargs = {'lr_lambda': lambda epoch: 1}
            
        scheduler = get_lr_scheduler(args.optimizer.lr_scheduler_name)
        scheduler_params, interval = get_lr_scheduler_params(args.optimizer.lr_scheduler_name, **scheduler_kwargs)
        scheduler = scheduler(optimizer, **scheduler_params)

        return [optimizer], [{'scheduler':scheduler, 'interval': interval, 'name': 'lr'}]