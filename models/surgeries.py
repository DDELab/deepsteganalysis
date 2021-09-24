import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from functools import partial
from operator import attrgetter
from torch import nn
import timm
import types
import torch
from models.models import zoo_params

    
def nostride(net):
    layer_name = zoo_params[net.model_name]['conv_stem_name']
    retriever = attrgetter(layer_name)
    retriever(net).stride = (1,1)
    return net

def luke_poststem(net): 
    num_channels = net._conv_stem.out_channels 
    net._conv_stem.stride = (1,1)
    
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