from typing import List
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import torch.distributed as dist
import torch.nn.functional as F
from torcheval.metrics import MulticlassAccuracy
from lightning import LightningModule
import wandb

from optim.optimizers import get_optimizer
from optim.schedulers import get_lr_scheduler
import models.surgeries
from models.models import get_net
from metrics.roc_metrics import wAUC, PE, MD5
from dataloading.decoders import decode_string    

class LitModel(LightningModule):
    """
    Train a steganalysis model
    """
    def __init__(self, args, in_chans, num_classes) -> None:
        
        self.args = args
        self.in_chans = in_chans
        self.num_classes = num_classes
        super().__init__()
        self.save_hyperparameters(self.args)
        
        self.train_metrics = {'train/PE': PE()}
        self.val_metrics = {'val/acc': MulticlassAccuracy(num_classes=2), 'val/wAUC': wAUC(), 'val/PE': PE(), 'val/MD5': MD5()}
        self.test_metrics = {'test/acc': MulticlassAccuracy(num_classes=2), 'test/wAUC': wAUC(), 'test/PE': PE(), 'test/MD5': MD5()}
        
        self.__set_attributes(self.train_metrics)
        self.__set_attributes(self.val_metrics)
        self.__set_attributes(self.test_metrics)
        self.__build_model()
    
    def __set_attributes(self, attributes_dict):
        for k,v in attributes_dict.items():
            setattr(self, k, v) 

    def __build_model(self):
        """Define model layers & loss."""
        # 1. Load pre-trained network:
        self.net = get_net(self.args.model.backbone, 
                           num_classes=self.num_classes,
                           in_chans=self.in_chans,
                           pretrained=self.args.ckpt.pretrained, 
                           ckpt_path=self.args.ckpt.seed_from)
        
        # 2. Do surgery if needed
        if self.args.model.surgery is not None:
            self.net = getattr(models.surgeries, self.args.model.surgery)(self.net)

        # 3. Loss:
        self.loss_func = F.cross_entropy

    def forward(self, x):
        """Forward pass. Returns logits."""

        x = self.net(x)
        
        return x

    def loss(self, logits, labels):
        return self.loss_func(logits, labels)

    def training_step(self, batch, batch_idx):
        # 1. Forward pass:
        x, y = batch
        y_logits = self.forward(x)
        
        # 2. Compute loss:
        train_loss = self.loss(y_logits, y)
            
        # 3. Compute metrics and log:
        self.log("train/loss", train_loss, on_step=True, on_epoch=False,  prog_bar=True, logger=False, sync_dist=False)
        for metric_name in self.train_metrics.keys():
            getattr(self, metric_name).update(y_logits, y)
            self.log(metric_name, getattr(self, metric_name).compute(), on_step=True, on_epoch=False, prog_bar=True, logger=False, sync_dist=False)

        return train_loss

    def on_train_epoch_end(self):
        for metric_name in self.train_metrics.keys():
            self.log(metric_name, getattr(self, metric_name).compute(), on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
            getattr(self, metric_name).reset()

    def validation_step(self, batch, batch_idx):
        # 1. Forward pass:
        x, y = batch
        y_logits = self.forward(x)

        # 2. Compute loss:
        val_loss = self.loss(y_logits, y)
        
        # 3. Compute metrics and log:
        self.log('val/loss', val_loss, on_step=True, on_epoch=False,  prog_bar=False, logger=False, sync_dist=False)
        for metric_name in self.val_metrics.keys():
            getattr(self, metric_name).update(y_logits, y)

    def on_validation_epoch_end(self):
        for metric_name in self.val_metrics.keys():
            self.log(metric_name, getattr(self, metric_name).compute(), on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
            getattr(self, metric_name).reset()
    
    def on_fit_end(self):
        super().on_fit_end()
        #TODO this is ugly
        if len(self.args.training.gpus or '') > 1:
            self.best_ckpt_path = [None]*len(self.args.training.gpus)
            dist.all_gather_object(self.best_ckpt_path, self.trainer.checkpoint_callback.best_model_path)
            self.best_ckpt_path = self.best_ckpt_path[0]
        else:
            self.best_ckpt_path = self.trainer.checkpoint_callback.best_model_path

    def on_test_epoch_start(self, *args, **kwargs):
        super().on_test_epoch_start(*args, **kwargs)
        self.test_table = wandb.Table(columns=['name', 'label'] + ['preds' + str(i) for i in range(self.num_classes)])

    def test_step(self, batch, batch_idx):
        # 1. Forward pass:
        x, y, name = batch
        y_logits = self.forward(x)

        # 2. Compute loss:
        tst_loss = self.loss(y_logits, y)
        for i in range(len(name)):
            self.test_table.add_data(decode_string(name[i]), y[i], *y_logits[i])
        
        # 3. Compute metrics and log:
        for metric_name in self.test_metrics.keys():
            getattr(self, metric_name).update(y_logits, y)

    def on_test_epoch_end(self):
        test_summary = {'best_ckpt_path': self.trainer.checkpoint_callback.best_model_path}
        for metric_name in self.test_metrics.keys():
            test_summary[metric_name] = getattr(self, metric_name).compute()
            getattr(self, metric_name).reset()
        if self.global_rank > 0:
            return
        for metric_name in self.test_metrics.keys():
            self.logger.experiment.summary[metric_name] = test_summary[metric_name]
        self.logger.experiment.log({'test_table': self.test_table})
        self.logger.experiment.summary['best_ckpt_path'] = test_summary['best_ckpt_path']
        return test_summary
        
    def configure_optimizers(self):
        param_list = list(self.net.named_parameters())

        optimizer = get_optimizer(param_list, self.args)

        train_len = len(self.trainer.datamodule.train_dataset)
        batch_size = self.args.training.batch_size
            
        scheduler, interval = get_lr_scheduler(optimizer, self.args, train_len, batch_size)
        return [optimizer], [{'scheduler':scheduler, 'interval': interval, 'name': 'lr'}]