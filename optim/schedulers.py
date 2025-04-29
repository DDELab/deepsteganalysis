import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, OneCycleLR, LambdaLR, MultiStepLR
      
def get_lr_scheduler(optimizer, args, train_len, batch_size):
    interval = 'epoch'
    if args.optimizer.lr_scheduler_name.lower() == 'lrdrop':
        # TODO: does this one work?
        scheduler = ReduceLROnPlateau(optimizer, mode='min',factor=0.5,
                                      patience=1,verbose=False, 
                                      threshold=0.0001,threshold_mode='abs',
                                      cooldown=0, min_lr=1e-8,
                                      eps=1e-08)
    elif args.optimizer.lr_scheduler_name.lower() == 'cos':
        interval = 'step'
        scheduler = CosineAnnealingLR(optimizer,
                                      T_max=args.training.epochs*train_len//len(args.training.gpus)//batch_size,
                                      eta_min=args.optimizer.lr/50)
    elif args.optimizer.lr_scheduler_name.lower()== 'onecycle':
        interval = 'step'
        scheduler = OneCycleLR(optimizer, 
                               max_lr=args.optimizer.lr, 
                               epochs=args.training.epochs,
                               steps_per_epoch=train_len//len(args.training.gpus)//batch_size,
                               pct_start=4.0/args.training.epochs,
                               div_factor=25,
                               final_div_factor=2)
    elif args.optimizer.lr_scheduler_name.lower()== 'multistep':
        scheduler = MultiStepLR(optimizer, 
                                milestones=[int(args.training.epochs*0.75)])
    elif args.optimizer.lr_scheduler_name.lower() == 'const':
        scheduler = LambdaLR(optimizer, 
                             lr_lambda=lambda epoch: 1)
    
    return scheduler, interval