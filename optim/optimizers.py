import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os 
import sys
import torch
import torch.nn.functional as F
      
def get_optimizer(param_list, args):
    if args.optimizer.decay_not_bias_norm:
        no_decay = ['bias', 'norm.bias', 'norm.weight', 'fc.weight', 'fc.bias']
    else:
        no_decay = []
    param_gorups = [
        {'params': [p for n, p in param_list if not any(nd in n for nd in no_decay)], 'weight_decay': args.optimizer.weight_decay},
        {'params': [p for n, p in param_list if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]      
    if args.optimizer.name.lower() == 'sgd':
        return torch.optim.SGD(param_gorups,  lr=args.optimizer.lr,  momentum=0.9)
    elif args.optimizer.name.lower() == 'adamw':
        return torch.optim.AdamW(param_gorups,  lr=args.optimizer.lr, eps=args.optimizer.eps)
    elif args.optimizer.name.lower() == 'adam':
        return torch.optim.Adam(param_gorups,  lr=args.optimizer.lr, eps=args.optimizer.eps)
    elif args.optimizer.name.lower() == 'adamax':
        return torch.optim.Adamax(param_gorups,  lr=args.optimizer.lr, eps=args.optimizer.eps)
