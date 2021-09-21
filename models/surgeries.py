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
    
    net._midstems = nn.ModuleList([timm.models.efficientnet_blocks.InvertedResidual(in_chs=num_channels, out_chs=num_channels, noskip=True),
                    timm.models.efficientnet_blocks.InvertedResidual(in_chs=num_channels, out_chs=num_channels),
                    timm.models.efficientnet_blocks.InvertedResidual(in_chs=num_channels, out_chs=num_channels),
                    timm.models.efficientnet_blocks.InvertedResidual(in_chs=num_channels, out_chs=num_channels, stride=2)])
    
    def new_extract_features(self, inputs):
        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))
        
        for idx, block in enumerate(self._midstems):
            x = block(x)
    
        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks) # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
    
        # Head
        x = self._swish(self._bn1(self._conv_head(x)))
        return x
    
    net.extract_features = types.MethodType(new_extract_features, net)
    return net