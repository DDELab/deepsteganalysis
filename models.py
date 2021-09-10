import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from functools import partial
from efficientnet_pytorch import EfficientNet
import numpy as np
from torch import nn
import timm
import numpy as np
import types
import torch
from senet import seresnet18, SEResNetBlock, SEResNetBlockNoRelu
from SRNet import SRNet

zoo_params = {
    
    'tf_efficientnetv2_m': {
        'fc_name': 'classifier',
        'fc': nn.Linear(in_features=1280, out_features=4, bias=True),
        'init_op': partial(timm.create_model, 'tf_efficientnetv2_m', pretrained=True, in_chans=1) 
    },
    
    'efficientnetv2_rw_m':{
        'fc_name': 'classifier',
        'fc': nn.Linear(in_features=2152, out_features=4, bias=True),
        'init_op': partial(timm.create_model, 'efficientnetv2_rw_m', pretrained=True, in_chans=1) 
    },

    'tf_efficientnetv2_l_jin':{
        'fc_name': 'classifier',
        'fc': nn.Linear(in_features=1280, out_features=4, bias=True),
        'init_op': partial(timm.create_model, 'tf_efficientnetv2_l', pretrained=False, in_chans=3) 
    },

    'tf_efficientnetv2_l_in21ft1k':{
        'fc_name': 'classifier',
        'fc': nn.Linear(in_features=1280, out_features=4, bias=True),
        'init_op': partial(timm.create_model, 'tf_efficientnetv2_l_in21ft1k', pretrained=True, in_chans=1) 
    },
    
    'tf_efficientnetv2_l_in21k':{
        'fc_name': 'classifier',
        'fc': nn.Linear(in_features=1280, out_features=4, bias=True),
        'init_op': partial(timm.create_model, 'tf_efficientnetv2_l_in21k', pretrained=True, in_chans=1) 
    },
    
    'tf_efficientnetv2_l':{
        'fc_name': 'classifier',
        'fc': nn.Linear(in_features=1280, out_features=4, bias=True),
        'init_op': partial(timm.create_model, 'tf_efficientnetv2_l', pretrained=True, in_chans=1) 
    },
    
    'tf_efficientnetv2_m_in21k':{
        'fc_name': 'classifier',
        'fc': nn.Linear(in_features=1280, out_features=4, bias=True),
        'init_op': partial(timm.create_model, 'tf_efficientnetv2_m_in21k', pretrained=True, in_chans=1) 
    },
    
    'srnet': {
        'fc_name': 'fc',
        'fc': nn.Linear(in_features=512, out_features=4, bias=True),
        'init_op': partial(SRNet, in_channels=1, nclasses=2)
    },
    
    'srnet_color': {
        'fc_name': 'fc',
        'fc': nn.Linear(in_features=512, out_features=4, bias=True),
        'init_op': partial(SRNet, in_channels=3, nclasses=4)
    },
    
    'seresnext26_32x4d': {
        'fc_name': 'last_linear',
        'fc': nn.Linear(in_features=2048, out_features=4, bias=True),
        'init_op': partial(timm.create_model, 'seresnext26_32x4d', pretrained=True)
    },
    
    'efficientnet-b0': {
        'fc_name': '_fc',
        'fc': nn.Linear(in_features=1280, out_features=2, bias=True),
        'init_op': partial(EfficientNet.from_pretrained, 'efficientnet-b0')
    },
    
    'efficientnet-b2': {
        'fc_name': '_fc',
        'fc': nn.Linear(in_features=1408, out_features=2, bias=True),
        'init_op': partial(EfficientNet.from_pretrained, 'efficientnet-b2')
    },
    
    'efficientnet-b4': {
        'fc_name': '_fc',
        'fc': nn.Linear(in_features=1792, out_features=4, bias=True),
        'init_op': partial(EfficientNet.from_pretrained, 'efficientnet-b4')
    },
    
    'efficientnet-b5': {
        'fc_name': '_fc',
        'fc': nn.Linear(in_features=2048, out_features=4, bias=True),
        'init_op': partial(EfficientNet.from_pretrained, 'efficientnet-b5')
    },
    
    'efficientnet-b6': {
        'fc_name': '_fc',
        'fc': nn.Linear(in_features=2304, out_features=4, bias=True),
        'init_op': partial(EfficientNet.from_pretrained, 'efficientnet-b6')
    },
    
    'mixnet_xl': {
        'fc_name': 'classifier',
        'fc': nn.Linear(in_features=1536, out_features=4, bias=True),
        'init_op': partial(timm.create_model, 'mixnet_xl', pretrained=True)
    },
    
    'mixnet_s': {
        'fc_name': 'classifier',
        'fc': nn.Linear(in_features=1536, out_features=2, bias=True),
        'init_op':  partial(timm.create_model, 'mixnet_s', pretrained=True)
    }, 
    
    'mixnet_s_fromscratch': {
        'fc_name': 'classifier',
        'fc': nn.Linear(in_features=1536, out_features=2, bias=True),
        'init_op':  partial(timm.create_model, 'mixnet_s', pretrained=False)
    }, 
    
    'seresnet18': {
        'fc_name': 'last_linear',
        'fc': nn.Linear(in_features=512, out_features=4, bias=True),
        'init_op': partial(seresnet18, pretrained=True, num_classes=4)
    }, 
}

def get_net(model_name):
    net = zoo_params[model_name]['init_op']()
    setattr(net, zoo_params[model_name]['fc_name'], zoo_params[model_name]['fc'])
    if 'tf_efficient' in model_name:
        return net
    if '_rw_' in model_name:
        return net
    if 'mixnet' in model_name:
        net.conv_stem = nn.Sequential(nn.Conv2d(1, 3, 1, stride=1, bias=False),
                                     nn.BatchNorm2d(3),
                                     nn.ReLU(inplace=True),
                                     net.conv_stem)
    elif 'efficient' in model_name:
        net._conv_stem = nn.Sequential(nn.Conv2d(1, 3, 1, stride=1, bias=False),
                                     nn.BatchNorm2d(3),
                                     nn.ReLU(inplace=True),
                                     net._conv_stem)   
    elif 'seresnet' in model_name:
        net.layer0 = nn.Sequential(nn.Conv2d(1, 3, 1, stride=1, bias=False),
                                     nn.BatchNorm2d(3),
                                     nn.ReLU(inplace=True),
                                     net.layer0)  
    return net
    
def nostride_xu(net):
    net.layer0[-1].conv1.stride = 1
    return net

def add_1x1_conv_srnet(net):
    net.block1 = nn.Sequential(nn.Conv2d(1, 3, 1, stride=1, bias=False),
                                     nn.BatchNorm2d(3),
                                     nn.ReLU(inplace=True),
                                     net.block1)
    return net
    
def nostride(net): 
    net._conv_stem[-1].stride = (1,1)
    return net
    
    
def surgery_b0midstem_resnetblock(net): 
    num_channels = net._conv_stem[-1].out_channels 
    net._conv_stem[-1].stride = (1,1)
    downsample = nn.Sequential(nn.Conv2d(num_channels, num_channels,
                                        kernel_size=1, stride=2,
                                        padding=0, bias=False),
                               nn.BatchNorm2d(num_channels))
    net._midstems = nn.ModuleList([SEResNetBlock(inplanes=num_channels, planes=num_channels, reduction=16, groups=1),
                                   SEResNetBlock(inplanes=num_channels, planes=num_channels, reduction=16, groups=1),
                                   SEResNetBlock(inplanes=num_channels, planes=num_channels, reduction=16, groups=1),
                                   SEResNetBlock(inplanes=num_channels, planes=num_channels, reduction=16, 
                                                 downsample=downsample, groups=1, stride=2)])
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
    
    

def surgery_b0midstem_resnetblock_nodoublerelu(net): 
    num_channels = net._conv_stem[-1].out_channels 
    net._conv_stem[-1].stride = (1,1)
    downsample = nn.Sequential(nn.Conv2d(num_channels, num_channels,
                                        kernel_size=1, stride=2,
                                        padding=0, bias=False),
                               nn.BatchNorm2d(num_channels))
    net._midstems = nn.ModuleList([SEResNetBlockNoRelu(inplanes=num_channels, planes=num_channels, reduction=16, groups=1),
                                   SEResNetBlockNoRelu(inplanes=num_channels, planes=num_channels, reduction=16, groups=1),
                                   SEResNetBlockNoRelu(inplanes=num_channels, planes=num_channels, reduction=16, groups=1),
                                   SEResNetBlockNoRelu(inplanes=num_channels, planes=num_channels, reduction=16, 
                                                 downsample=downsample, groups=1, stride=2)])
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
    
    

def surgery_b0midstem_resnetblock_3(net): 
    num_channels = net._conv_stem[-1].out_channels 
    net._conv_stem[-1].stride = (1,1)
    downsample = nn.Sequential(nn.Conv2d(num_channels, num_channels,
                                        kernel_size=1, stride=2,
                                        padding=0, bias=False),
                               nn.BatchNorm2d(num_channels))
    net._midstems = nn.ModuleList([SEResNetBlock(inplanes=num_channels, planes=num_channels, reduction=16, groups=1),
                                   SEResNetBlock(inplanes=num_channels, planes=num_channels, reduction=16, groups=1),
                                   SEResNetBlock(inplanes=num_channels, planes=num_channels, reduction=16, 
                                                 downsample=downsample, groups=1, stride=2)])
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
    
    
def surgery_b0midstem_resnetblock_nodoublerelu_3(net): 
    num_channels = net._conv_stem[-1].out_channels 
    net._conv_stem[-1].stride = (1,1)
    downsample = nn.Sequential(nn.Conv2d(num_channels, num_channels,
                                        kernel_size=1, stride=2,
                                        padding=0, bias=False),
                               nn.BatchNorm2d(num_channels))
    net._midstems = nn.ModuleList([SEResNetBlockNoRelu(inplanes=num_channels, planes=num_channels, reduction=16, groups=1),
                                   SEResNetBlockNoRelu(inplanes=num_channels, planes=num_channels, reduction=16, groups=1),
                                   SEResNetBlockNoRelu(inplanes=num_channels, planes=num_channels, reduction=16, 
                                                 downsample=downsample, groups=1, stride=2)])
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

def surgery_lum_nostride(net): 
    net._conv_stem.stride = (1,1)
    net._conv_stem = nn.Sequential(nn.Conv2d(1, 3, 1, stride=1, bias=False),
                                     nn.BatchNorm2d(3),
                                     nn.ReLU(inplace=True),
                                     net._conv_stem)
        
    return net


def surgery_b0midstem(net): 
    num_channels = 32
    net._conv_stem.stride = (1,1)
    net._conv_stem = nn.Sequential(nn.Conv2d(1, 3, 1, stride=1, bias=False),
                                     nn.BatchNorm2d(3),
                                     nn.ReLU(inplace=True),
                                     net._conv_stem)
    
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

def surgery_seresnet(net): 
    net.layer0.pool = nn.Identity()
    #net.prelayer = nn.Sequential(nn.Conv2d(3, 6, 3, stride=1, padding=3, bias=False),
    #                             nn.BatchNorm2d(6),
    #                             nn.ReLU(inplace=True),
    #                             nn.Conv2d(6, 12, 3, stride=1, padding=3, bias=False),
    #                             nn.BatchNorm2d(12),
    #                             nn.ReLU(inplace=True),
    #                             nn.Conv2d(12, 36, 3, stride=1, padding=3, bias=False),
    #                             nn.BatchNorm2d(36),
    #                             nn.ReLU(inplace=True),
    #                            )
    #
    #def new_forward_features(self, x):
    #    x = self.prelayer(x)
    #    x = self.layer0(x)
    #    x = self.layer1(x)
    #    x = self.layer2(x)
    #    x = self.layer3(x)
    #    x = self.layer4(x)
    #    return x
    #
    #net.forward_features = types.MethodType(new_forward_features, net)
    #net.layer0.conv1.weight = nn.Parameter(net.layer0.conv1.weight.repeat(1, 12, 1, 1))
    
    return net


def surgery_nostride(net):
    net._conv_stem.stride = (1,1)
    return net
    
    
def surgery_seresnext(net):
    net.drop_rate = 0.0
    net.layer0.pool = nn.Identity()

    #net.prelayer = nn.Sequential(nn.Conv2d(3, 6, 3, stride=1, padding=3, bias=False),
    #                             nn.BatchNorm2d(6),
    #                             nn.ReLU(inplace=True),
    #                             nn.Conv2d(6, 12, 3, stride=1, padding=3, bias=False),
    #                             nn.BatchNorm2d(12),
    #                             nn.ReLU(inplace=True),
    #                             nn.Conv2d(12, 36, 3, stride=1, padding=3, bias=False),
    #                             nn.BatchNorm2d(36),
    #                             nn.ReLU(inplace=True),
    #                            )
    #
    #def new_forward_features(self, x):
    #    x = self.prelayer(x)
    #    x = self.layer0(x)
    #    x = self.layer1(x)
    #    x = self.layer2(x)
    #    x = self.layer3(x)
    #    x = self.layer4(x)
    #    return x
    #
    #net.forward_features = types.MethodType(new_forward_features, net)
    #
    #net.layer0.conv1.weight = nn.Parameter(net.layer0.conv1.weight.repeat(1, 12, 1, 1))
    
    return net


def surgery_seresnet0(net): 
    
    net.prelayer = nn.Sequential(nn.Conv2d(3, 6, 3, stride=1, padding=3, bias=False),
                                 nn.BatchNorm2d(6),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(6, 12, 3, stride=1, padding=3, bias=False),
                                 nn.BatchNorm2d(12),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(12, 36, 3, stride=1, padding=3, bias=False),
                                 nn.BatchNorm2d(36),
                                 nn.ReLU(inplace=True),
                                )
    
    def new_forward_features(self, x):
        x = self.prelayer(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
    
    net.forward_features = types.MethodType(new_forward_features, net)
    
    net.layer0.conv1.weight = nn.Parameter(net.layer0.conv1.weight.repeat(1, 12, 1, 1))
    
    return net



def surgery_b0(net): 
    
    net.prelayer = nn.Sequential(nn.Conv2d(3, 6, 3, stride=1, padding=1, bias=False),
                                 nn.BatchNorm2d(6),
                                 nn.ReLU6(inplace=True),
                                 nn.Conv2d(6, 12, 3, stride=1, padding=1, bias=False),
                                 nn.BatchNorm2d(12),
                                 nn.ReLU6(inplace=True),
                                 nn.Conv2d(12, 36, 3, stride=1, padding=1, bias=False),
                                 nn.BatchNorm2d(36),
                                 nn.ReLU6(inplace=True))
    
    def new_extract_features(self, inputs):
        # Stem
        x = self.prelayer(inputs)
        x = self._swish(self._bn0(self._conv_stem(x)))
    
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
    
    net._conv_stem.weight = nn.Parameter(net._conv_stem.weight.repeat(1, 12, 1, 1))
    
    return net



#