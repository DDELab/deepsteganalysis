"""
Runs a model on a single node across multiple gpus.
"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import pandas as pd
from pathlib import Path
from argparse import ArgumentParser
from LitModel import *
from retriever import *
from torch.utils.data import DataLoader
import torch
import shutil
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateLogger
from pytorch_lightning.utilities.distributed import rank_zero_only

seed_everything(1994)

def setup_callbacks_loggers(args):
    
    log_path = Path('/home/ekaziak1/LogFiles/test/BOSS_alaska_stego/')
    name = args.backbone
    version = args.version
    if args.resume_from_checkpoint is not None:
        version = args.resume_from_checkpoint.split('/checkpoints')[0].split('/')[-1]
    tb_logger = TensorBoardLogger(log_path, name=name, version=version)
    lr_logger = LearningRateLogger(logging_interval='epoch')
    ckpt_callback = ModelCheckpoint(filepath=Path(tb_logger.log_dir)/'checkpoints/{epoch:02d}_{val_wAUC:.4f}', 
                                    save_top_k=3, save_last=True)
   
    return ckpt_callback, tb_logger, lr_logger


def main(args):
    """ Main training routine specific for this project. """
    
    model = LitModel(**vars(args))
    
    ckpt_callback, tb_logger, lr_logger = setup_callbacks_loggers(args)
    
    trainer = Trainer(checkpoint_callback=ckpt_callback,
                     logger=tb_logger,
                     callbacks=[lr_logger],
                     gpus=args.gpus,
                     min_epochs=args.epochs,
                     max_epochs=args.epochs,
                     precision=16,
                     amp_backend='apex',
                     amp_level='O1',
                     row_log_interval=100,
                     log_save_interval=100,
                     distributed_backend='ddp',
                     benchmark=True,
                     sync_batchnorm=True,
                     resume_from_checkpoint=args.resume_from_checkpoint)
    
    
    qfs = [args.qf]
    classes = [ ['QF'+str(q)+'/COVER', 'QF'+str(q)+'/JUNI_0.4_bpnzac'  ,'QF'+str(q)+'/UERD_0.2_bpnzac' ,'QF'+str(q)+'/J_MiPOD_0.4_bpnzac' ] for q in qfs ]
    
    IL_test = os.listdir(args.data_path+'QF75/COVER/TST/')
    
    dataset = []
    for cl in classes:
        for label, kind in enumerate(cl):
            for path in IL_test:
                dataset.append({
                    'kind': kind+'/TST',
                    'image_name': path,
                    'label': label,
                    'fold':2,
                })
    dataset = pd.DataFrame(dataset)

    test_dataset = TrainRetriever(
            data_path=args.data_path,
            kinds=dataset[dataset['fold'] == 2].kind.values,
            image_names=dataset[dataset['fold'] == 2].image_name.values,
            labels=dataset[dataset['fold'] == 2].label.values,
            transforms=get_valid_transforms(),
            decoder='NR',
            return_name=True,
            num_classes=len(classes[0])
        )
    
    test_dataloader = DataLoader(dataset=test_dataset,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            shuffle=False)

        
    trainer.test(model, test_dataloader, args.resume_from_checkpoint)

    logdir = Path(args.resume_from_checkpoint).parents[1]
    for g in range(len(args.gpus)):
        shutil.copy('predictions_rank_'+str(g)+'.txt', logdir/('predictions_rank_'+str(g)+'_QF'+str(args.qf)+'.txt'))


def run_cli():
    root_dir = os.path.dirname(os.path.realpath(__file__))
    
    parent_parser = ArgumentParser(add_help=False)

    parser = LitModel.add_model_specific_args(parent_parser)
    
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
    
    args = parser.parse_args()

    main(args)
    
    
if __name__ == '__main__':
    run_cli()