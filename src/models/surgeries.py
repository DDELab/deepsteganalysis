import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from functools import partial
from torch import nn
import timm
import types
import torch
from models.models import zoo_params

    
def nostride(net):
    assert 'efficientnet' in net.model_name, 'No stride only supported for EfficientNet'
    getattr(net, zoo_params[net.model_name]['conv_stem_name']).stride = (1,1)
    return net
        
def poststem(net): 
    assert 'efficientnet' in net.model_name, 'Post stem only supported for EfficientNet'
    num_channels = getattr(net, zoo_params[net.model_name]['conv_stem_name']).out_channels 
    net = nostride(net)
    
    net.post_stem = nn.ModuleList([timm.models.efficientnet_blocks.InvertedResidual(in_chs=num_channels, out_chs=num_channels, noskip=True),
                    timm.models.efficientnet_blocks.InvertedResidual(in_chs=num_channels, out_chs=num_channels),
                    timm.models.efficientnet_blocks.InvertedResidual(in_chs=num_channels, out_chs=num_channels),
                    timm.models.efficientnet_blocks.InvertedResidual(in_chs=num_channels, out_chs=num_channels, stride=2)])
    
    def new_forward_features(self, x):
        # Stem
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        for idx, block in enumerate(self.post_stem):
            x = block(x)
        x = self.blocks(x)
        x = self.conv_head(x)
        x = self.bn2(x)
        x = self.act2(x)
        return x
    
    net.forward_features = types.MethodType(new_forward_features, net)
    return net