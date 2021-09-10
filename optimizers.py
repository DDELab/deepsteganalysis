import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os 
import sys
import torch
import torch.nn.functional as F
      
def get_optimizer(optimizer_name):
    if optimizer_name.lower() == 'sgd':
        return torch.optim.SGD
    elif optimizer_name.lower() == 'adamw':
        return torch.optim.AdamW
    elif optimizer_name.lower() == 'adam':
        return torch.optim.Adam
    elif optimizer_name.lower() == 'adamax':
        return torch.optim.Adamax

    
def get_lr_scheduler(scheduler_name):
    if scheduler_name.lower() == 'lrdrop':
        return torch.optim.lr_scheduler.ReduceLROnPlateau
    elif scheduler_name.lower() == 'cos':
        return torch.optim.lr_scheduler.CosineAnnealingLR
    elif scheduler_name.lower() == 'onecycle':
        return torch.optim.lr_scheduler.OneCycleLR
    elif scheduler_name.lower() == 'multistep':
        return torch.optim.lr_scheduler.MultiStepLR
    elif scheduler_name.lower() == 'const':
        return torch.optim.lr_scheduler.LambdaLR
    
    
def get_lr_scheduler_params(scheduler_name, **kwargs):
    
    if scheduler_name.lower() == 'lrdrop':
        params = dict(
                mode='min',
                factor=0.5,
                patience=1,
                verbose=False, 
                threshold=0.0001,
                threshold_mode='abs',
                cooldown=0, 
                min_lr=1e-8,
                eps=1e-08
            )
        interval = 'epoch'
        return params, interval
    
    elif scheduler_name.lower() in {'cos', 'onecycle'}: 
        params = kwargs
        interval = 'step'
        return params, interval
    
    elif scheduler_name.lower() in {'multistep', 'const'}:
        interval = 'epoch'
        params = kwargs
        return params, interval
    
    
class LabelSmoothing(torch.nn.Module):
    def __init__(self, smoothing = 0.05):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        if self.training:
            x = x.float()
            target = target.float()
            logprobs = F.log_softmax(x, dim = -1)

            nll_loss = -logprobs * target
            nll_loss = nll_loss.sum(-1)
    
            smooth_loss = -logprobs.mean(dim=-1)
            loss = self.confidence * nll_loss + self.smoothing * smooth_loss
            return loss.mean()
        else:
            return F.cross_entropy(x, target)