"""
Runs a model on a single node across multiple gpus.
"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
from pathlib import Path
from argparse import ArgumentParser

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.utilities.distributed import rank_zero_only

from LitModel import LitModel
from dataloading.LitDataloader import LitStegoDataModule
from tools.log_utils import gen_run_name, setup_callbacks_loggers
from tools.options_utils import get_args_cli_yaml

def main(args):
    """ Main training routine specific for this project. """

    seed_everything(args.training.seed)

    model = LitModel(args)
    datamodule = LitStegoDataModule(args)
    ckpt_callback, loggers, lr_logger = setup_callbacks_loggers(args)

    trainer = Trainer(logger=loggers,
                      callbacks=[ckpt_callback, lr_logger],
                      gpus=args.training.gpus,
                      min_epochs=args.training.epochs,
                      max_epochs=args.training.epochs,
                      precision=16,
                      amp_backend='native',
                      amp_level=args.training.amp_level,
                      log_every_n_steps=100,
                      flush_logs_every_n_steps=100,
                      accelerator='ddp' if len(args.training.gpus) > 1 else None,
                      benchmark=True,
                      sync_batchnorm=True,
                      resume_from_checkpoint=args.ckpt.resume_from)

    trainer.logger.log_hyperparams(model.hparams)

    trainer.fit(model, datamodule)

def run_cli():
    # os.path.join(os.path.expanduser('~')
    args = get_args_cli_yaml(cfg_path="cfg/debug.yaml")
    main(args)

if __name__ == '__main__':
    run_cli()
