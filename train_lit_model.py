"""
Runs a model on a single node across multiple gpus.
"""
import warnings

from torch.utils.data import dataloader
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
from pathlib import Path
from argparse import ArgumentParser
import torch
import uuid

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.utilities.distributed import rank_zero_only

from LitModel import LitModel
from LitDataloader import LitStegoDataModule

seed_everything(1994)

def setup_callbacks_loggers(args):
    
    log_path = Path('/home/ekaziak1/LogFiles/test/BOSS_alaska_stego/')
    name = args.backbone
    version = args.version
    
    if not args.all_qfs:
        log_path = log_path/('QF'+args.qf)
    else:
        log_path = log_path/('all_qfs')
    
    wandb_logger = WandbLogger(project='imagenet-steganalysis-test',
                               entity='dde',
                               config=args, name=args.run_name,
                               mode='online' if args.wb==True else 'offline')
    tb_logger = TensorBoardLogger(log_path, name=name, version=version)
    lr_logger = LearningRateMonitor(logging_interval='epoch')
    ckpt_callback = ModelCheckpoint(dirpath='logpath/version/checkpoints',
                                    filename='{epoch:02d}_{val_wAUC:.4f}', 
                                    save_top_k=5, save_last=True, monitor='val_wAUC', mode='max')
   
    return ckpt_callback, wandb_logger, lr_logger


def main(args):
    """ Main training routine specific for this project. """
    
    if args.seed_from_checkpoint:
        print('model seeded')
        model = LitModel.load_from_checkpoint(args.seed_from_checkpoint, **vars(args), strict=False)
    else:
        model = LitModel(**vars(args))

    datamodule = LitStegoDataModule(**vars(args))
    
    ckpt_callback, tb_logger, lr_logger = setup_callbacks_loggers(args)
    
    trainer = Trainer(logger=tb_logger,
                      callbacks=[ckpt_callback, lr_logger],
                      gpus=args.gpus,
                      min_epochs=args.epochs,
                      max_epochs=args.epochs,
                      precision=16,
                      amp_backend='native',
                      amp_level='O1',
                      log_every_n_steps=100,
                      flush_logs_every_n_steps=100,
                      distributed_backend='ddp' if len(args.gpus) > 1 else None,
                      benchmark=True,
                      sync_batchnorm=True,
                      resume_from_checkpoint=args.resume_from_checkpoint)
    
    trainer.logger.log_hyperparams(model.hparams)
    
    trainer.fit(model, datamodule)


def run_cli():
    root_dir = os.path.dirname(os.path.realpath(__file__))
    
    parent_parser = ArgumentParser(add_help=False)

    parser = LitModel.add_model_specific_args(parent_parser)
    parser = LitStegoDataModule.add_data_specific_args(parser)
    
    parser.add_argument('--version',
                         default=None,
                         type=str,
                         metavar='V',
                         help='version or id of the net')
    parser.add_argument('--resume-from-checkpoint',
                         default=None,
                         type=str,
                         metavar='RFC',
                         help='path to checkpoint')
    parser.add_argument('--seed-from-checkpoint',
                         default=None,
                         type=str,
                         metavar='SFC',
                         help='path to checkpoint seed')
    parser.add_argument('--run-name',
                        default=str(uuid.uuid4())[:8],
                        type=str,
                        help='run name for weight&biases entry')
    parser.add_argument('--wb',
                        default=False,
                        type=bool,
                        help='turn on/off W&B logging')
    
    parser = ArgumentParser(parents=[parser])

    args = parser.parse_args()
    main(args)


if __name__ == '__main__':
    run_cli()