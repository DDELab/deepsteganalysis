import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from functools import partial
import numpy as np
from torch import nn
import timm
import numpy as np
import types
import torch
from models.SRNet import SRNet

zoo_params = {

    'eca_nfnet_l1': {
        'fc_name': 'fc',
        'conv_stem_name': 'stem.conv1',
        'init_op': partial(timm.create_model, 'eca_nfnet_l1') 
    },
    
    'efficientnet_b2': {
        'fc_name': 'classifier',
        'conv_stem_name': 'conv_stem',
        'init_op': partial(timm.create_model, 'efficientnet_b2') 
    },

    'efficientnet_b4': {
        'fc_name': 'classifier',
        'conv_stem_name': 'conv_stem',
        'init_op': partial(timm.create_model, 'efficientnet_b4') 
    },

    'srnet': {
        'fc_name': 'fc',
        'conv_stem_name': 'block1.0.conv',
        'init_op': SRNet
    },

}

def get_net(model_name, num_classes=2, in_chans=3, imagenet=True, ckpt_path=None, strict_loading=False):
    net = zoo_params[model_name]['init_op'](num_classes=num_classes, in_chans=in_chans, pretrained=imagenet)
    net.model_name = model_name

    if ckpt_path is not None:
        state_dict = torch.load(ckpt_path)['state_dict']
        state_dict = {k.split('net.')[1]: v for k, v in state_dict.items()}
        
        # Check FC compatibility
        out_fc, _ = state_dict[zoo_params[model_name]['fc_name'] + '.weight'].shape
        if out_fc != num_classes:
            del state_dict[zoo_params[model_name]['fc_name'] + '.weight']
            del state_dict[zoo_params[model_name]['fc_name'] + '.bias']
        
        # Check first conv
        weight_name = zoo_params[model_name]['conv_stem_name'] + '.weight'
        _,in_conv,_,_ = state_dict[weight_name].shape
        if in_conv != in_chans:
            state_dict[weight_name] = timm.models.helpers.adapt_input_conv(in_chans, state_dict[weight_name])
        
        net.load_state_dict(state_dict, strict=strict_loading)
    return net