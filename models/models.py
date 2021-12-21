import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from functools import partial
from validators import url
import numpy as np
from torch import nn
import timm
import numpy as np
import types
import torch
from models.srnet import SRNet
from models.manifold_srnet import ManifoldSRNet
from models.onehotconv import OneHotConv

zoo_params = {

    'efficientnet_b2': {
        'fc_name': 'classifier',
        'conv_stem_name': ['conv_stem'],
        'init_op': partial(timm.create_model, 'efficientnet_b2')
    },

    'efficientnet_b4': {
        'fc_name': 'classifier',
        'conv_stem_name': ['conv_stem'],
        'init_op': partial(timm.create_model, 'efficientnet_b4')
    },

    'srnet': {
        'fc_name': 'fc',
        'conv_stem_name': ['block1.0.conv'],
        'init_op': SRNet
    },

    'manifold_srnet': {
        'fc_name': 'fc',
        'conv_stem_name': ['block1.0.conv'],
        'init_op': ManifoldSRNet
    },

    'onehotconv': {
        'fc_name': 'fc',
        'conv_stem_name': ['layer1.conv', 'layer1.conv_dilated'],
        'init_op': OneHotConv
    }

}

def get_net(model_name, num_classes=2, in_chans=3, pretrained=True, ckpt_path=None, strict_loading=False):
    net = zoo_params[model_name]['init_op'](num_classes=num_classes, in_chans=in_chans, pretrained=pretrained)
    net.model_name = model_name

    if ckpt_path is not None:
        if url(ckpt_path):
            state_dict = torch.hub.load_state_dict_from_url(ckpt_path)['state_dict']
        else:
            state_dict = torch.load(ckpt_path)['state_dict']
        state_dict = {k.split('net.')[1]: v for k, v in state_dict.items()}

        # Check FC compatibility
        #out_fc, _ = state_dict[zoo_params[model_name]['fc_name'] + '.weight'].shape
        #if out_fc != num_classes:
        #    del state_dict[zoo_params[model_name]['fc_name'] + '.weight']
        #    del state_dict[zoo_params[model_name]['fc_name'] + '.bias']
#
        ## Check first convs
        #for conv_stem_name in zoo_params[model_name]['conv_stem_name']:
        #    weight_name =  conv_stem_name + '.weight'
        #    _,in_conv,_,_ = state_dict[weight_name].shape
        #    if in_conv != in_chans:
        #        state_dict[weight_name] = timm.models.helpers.adapt_input_conv(in_chans, state_dict[weight_name])

        net.load_state_dict(state_dict, strict=strict_loading)

        # clean up
        del state_dict
    return net
