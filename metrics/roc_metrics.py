"""
TODO: write these in pytorch to get rid of the numpy conversion decorators
"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.metrics import roc_curve, auc, roc_auc_score
import numpy as np
import torch.nn.functional as F
import torch
from tools.decorators import _numpy_metric_conversion
from tools.numpy_utils import check_nans
from torchmetrics.utilities.data import dim_zero_cat
from torcheval.metrics import AUC

@_numpy_metric_conversion
def roc_auc_score_np(x, y):
    return roc_auc_score(x, y)

@_numpy_metric_conversion
def wauc(y_true, y_pred):
    """
    https://www.kaggle.com/anokas/weighted-auc-metric-updated
    """
    tpr_thresholds = [0.0, 0.4, 1.0]
    weights = [2, 1]
    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1, drop_intermediate=False)
    if check_nans(fpr, tpr):
        return np.nan
    areas = np.array(tpr_thresholds[1:]) - np.array(tpr_thresholds[:-1])
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
        competition_metric += submetric

    return competition_metric/normalization\
        
@_numpy_metric_conversion
def md5(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1, drop_intermediate=False)
    if check_nans(fpr, tpr):
        return np.nan
    return 1-np.interp(0.05, fpr, tpr) 

@_numpy_metric_conversion   
def pe(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1, drop_intermediate=False)
    if check_nans(fpr, tpr):
        return np.nan
    P = 0.5*(fpr+(1-tpr))
    return min(P[P>0])

class wAUC(AUC):
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:  # type: ignore
        """Update state with predictions and targets.
        """
        #preds = 1 - F.softmax(preds.double(), dim=1)[:,0]
        target = torch.clip(target, min=0, max=1)
        self.x.append(preds[:,1])
        self.y.append(target)
    def compute(self):
        """Computes MD5 based on inputs passed in to ``update`` previously."""
        x = dim_zero_cat(self.x)
        y = dim_zero_cat(self.y)
        return wauc(y, x)

class AUC(AUC):
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:  # type: ignore
        """Update state with predictions and targets.
        """
        #preds = 1 - F.softmax(preds.double(), dim=1)[:,0]
        target = torch.clip(target, min=0, max=1)
        self.x.append(preds[:,1])
        self.y.append(target)
    def compute(self):
        """Computes MD5 based on inputs passed in to ``update`` previously."""
        x = dim_zero_cat(self.x)
        y = dim_zero_cat(self.y)
        return roc_auc_score_np(y, x)

class MD5(AUC):
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:  # type: ignore
        """Update state with predictions and targets.
        """
        #preds = 1 - F.softmax(preds.double(), dim=1)[:,0]
        target = torch.clip(target, min=0, max=1)
        #self.x.append(preds)
        self.x.append(preds[:,1])
        self.y.append(target)
    def compute(self):
        """Computes MD5 based on inputs passed in to ``update`` previously."""
        x = dim_zero_cat(self.x)
        y = dim_zero_cat(self.y)
        return md5(y, x)

class PE(AUC):
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:  # type: ignore
        """Update state with predictions and targets.
        """
        #preds = 1 - F.softmax(preds.double(), dim=1)[:,0]
        target = torch.clip(target, min=0, max=1)
        #self.x.append(preds)
        self.x.append(preds[:,1])
        self.y.append(target)

    def compute(self):
        """Computes PE based on inputs passed in to ``update`` previously."""
        x = dim_zero_cat(self.x)
        y = dim_zero_cat(self.y)
        return pe(y, x)
