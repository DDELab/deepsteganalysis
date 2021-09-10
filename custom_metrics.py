import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.metrics import roc_curve, auc
import numpy as np
import torch.nn.functional as F
import torch
from pytorch_lightning.metrics.metric import NumpyMetric
from pytorch_lightning.metrics.converters import sync_ddp, _numpy_metric_conversion
from pytorch_lightning.core.decorators import auto_move_data


@_numpy_metric_conversion
def wauc(y_true, y_valid):
    """
    https://www.kaggle.com/anokas/weighted-auc-metric-updated
    """
    tpr_thresholds = [0.0, 0.4, 1.0]
    weights = [2, 1]
    y_true = y_true.clip(min=0, max=1).astype(int)
    fpr, tpr, thresholds = roc_curve(y_true, y_valid, pos_label=1)

    # size of subsets
    areas = np.array(tpr_thresholds[1:]) - np.array(tpr_thresholds[:-1])

    # The total area is normalized by the sum of weights such that the final weighted AUC is between 0 and 1.
    normalization = np.dot(areas, weights)

    competition_metric = 0
    for idx, weight in enumerate(weights):
        y_min = tpr_thresholds[idx]
        y_max = tpr_thresholds[idx + 1]
        mask = (y_min < tpr) & (tpr < y_max)
        if mask.sum()==0:
            continue 
        x_padding = np.linspace(fpr[mask][-1], 1, 100)
        x = np.concatenate([fpr[mask], x_padding])
        y = np.concatenate([tpr[mask], [y_max] * len(x_padding)])
        y = y - y_min  
        score = auc(x, y)
        submetric = score * weight
        best_subscore = (y_max - y_min) * weight
        competition_metric += submetric

    return competition_metric/normalization


class RocAucMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.y_true = torch.tensor([0,1])
        self.y_pred = torch.tensor([0.5,0.5])
        self.score = 0

    def update(self, y_pred, y_true):
        y_true = torch.argmax(y_true, dim=1)
        y_pred = 1 - F.softmax(y_pred.double(), dim=1)[:,0]
        self.y_true = torch.cat((self.y_true, y_true.detach().cpu()), 0)
        self.y_pred = torch.cat((self.y_pred, y_pred.detach().cpu()), 0)
        self.score = wauc(self.y_true, self.y_pred)
        return self.score 
    
    #@sync_ddp(reduce_op='avg')
    def avg(self):
        return self.score.cuda()
        
        
@_numpy_metric_conversion
def md5(y_true, y_valid):
    y_true = y_true.clip(min=0, max=1).astype(int)
    fpr, tpr, thresholds = roc_curve(y_true, y_valid, pos_label=1)
    return 1-np.interp(0.05, fpr, tpr) 

@_numpy_metric_conversion   
def pe(y_true, y_valid):
    y_true = y_true.clip(min=0, max=1).astype(int)
    fpr, tpr, thresholds = roc_curve(y_true, y_valid, pos_label=1)
    P = 0.5*(fpr+(1-tpr))
    return  min(P[P>0]) # 1 minus

class MD5Meter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.y_true = torch.tensor([0,1])
        self.y_pred = torch.tensor([0.5,0.5])
        self.score = 0

    def update(self, y_pred, y_true):
        y_true = torch.argmax(y_true, dim=1)
        y_pred = 1 - F.softmax(y_pred.double(), dim=1)[:,0]
        self.y_true = torch.cat((self.y_true, y_true.detach().cpu()), 0)
        self.y_pred = torch.cat((self.y_pred, y_pred.detach().cpu()), 0)
        self.score = md5(self.y_true, self.y_pred)
        return self.score 
    
    #@sync_ddp(reduce_op='avg')
    def avg(self):
        return self.score.cuda()
        

class PEMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.y_true = torch.tensor([0,1])
        self.y_pred = torch.tensor([0.5,0.5])
        self.score = 0

    def update(self, y_pred, y_true):
        y_true = torch.argmax(y_true, dim=1)
        y_pred = 1 - F.softmax(y_pred.double(), dim=1)[:,0]
        self.y_true = torch.cat((self.y_true, y_true.detach().cpu()), 0)
        self.y_pred = torch.cat((self.y_pred, y_pred.detach().cpu()), 0)
        self.score = pe(self.y_true, self.y_pred)
        return self.score 
    
    #@sync_ddp(reduce_op='avg')
    def avg(self):
        return self.score.cuda()
    
