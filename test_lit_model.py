"""
Runs a model on a single node across multiple gpus.
"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from lightning import Trainer, seed_everything
import os

from LitModel import LitModel
from dataloading.LitDataloader import LitStegoDataModule
from tools.log_utils import setup_callbacks_loggers
from tools.options_utils import get_args_cli_yaml

def main(args):
    """ Main training routine specific for this project. """

    seed_everything(args.training.seed)
    
    datamodule = LitStegoDataModule(args)
    model = LitModel(args, datamodule.in_chans, datamodule.num_classes)
    ckpt_callback, loggers, lr_logger = setup_callbacks_loggers(args)

    trainer = Trainer(logger=loggers,
                      callbacks=[ckpt_callback, lr_logger],
                      devices=args.training.gpus if args.training.gpus else os.cpu_count()//2,
                      min_epochs=args.training.epochs,
                      max_epochs=args.training.epochs,
                      precision=str(args.training.precision),
                      log_every_n_steps=200,
                      accelerator='gpu' if args.training.gpus else 'cpu',
                      benchmark=True,
                      inference_mode=True,
                      sync_batchnorm=len(args.training.gpus or '') > 1)

    ## saves predictions as a table in artifacts
    trainer.test(model, datamodule, ckpt_path=args.ckpt.resume_from)

def run_cli():
    # os.path.join(os.path.expanduser('~')
    args = get_args_cli_yaml(cfg_path="cfg/default.yaml")
    main(args)

if __name__ == '__main__':
    run_cli()
