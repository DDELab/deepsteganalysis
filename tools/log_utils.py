import uuid
import os 
import random
from wonderwords import RandomWord

from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.utilities.distributed import _get_rank


def gen_run_name():
    """
    Adapted from https://github.com/wandb/lit_utils/blob/main/utils.py
    """
    r = RandomWord()
    name = "-".join(
            [r.word(word_min_length=3, word_max_length=8, include_parts_of_speech=["adjective"]) ,
            r.word(word_min_length=3, word_max_length=8, include_parts_of_speech=["noun"]),
            str(random.randint(0,9))
            ])

    return name

def setup_callbacks_loggers(args):
    
    if args.logging.project is None:
        args.logging.project = os.getcwd().split('/')[-1]

    if args.logging.eid is None:
        args.logging.eid = gen_run_name()

    args.logging.path = os.path.join(os.path.expanduser("~"), args.logging.path)
    
    log_path = os.path.join(args.logging.path, args.logging.project,
                            args.model.backbone)
    
    log_path = os.path.join(log_path, 'all_qfs')

    log_path = os.path.join(log_path, args.logging.eid)
    tb_dir = os.path.join(log_path, "tensorboard")
    if _get_rank() == 0:
        os.makedirs(tb_dir, exist_ok=True)
    
    wandb_logger = WandbLogger(project=args.logging.project,
                               dir=log_path,         
                               name=args.logging.eid,
                               entity=args.logging.wandb.team,
                               mode='online' if args.logging.wandb.activate==True else 'offline')

    tb_logger = TensorBoardLogger(tb_dir, name="", version="")
    lr_logger = LearningRateMonitor(logging_interval='epoch')
    ckpt_callback = ModelCheckpoint(dirpath=os.path.join(log_path, 'checkpoints'),
                                    filename='{epoch:02d}_{val_wAUC:.4f}', 
                                    save_top_k=5, save_last=True, monitor='val_wAUC', mode='max')
   
    return ckpt_callback, [wandb_logger, tb_logger], lr_logger