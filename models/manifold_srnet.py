'''
"Deep Residual Network for Steganalysis of Digital Images"
Mehdi Boroumand, Mo Chen, and Jessica Fridrich
http://www.ws.binghamton.edu/fridrich/Research/SRNet.pdf
'''
import torch
import types
from timm.models.layers import Swish
from timm.models.layers.adaptive_avgmax_pool import SelectAdaptivePool2d
from timm.models.layers import create_conv2d, create_pool2d
import timm
from torch import nn
import numpy as np
from functools import partial
import warnings
from models.convolutional_filter_manifold import ConvolutionalFilterManifold
warnings.simplefilter(action='ignore', category=FutureWarning)
# from timm.models.layers.activations_me import SwishMe
from timm.models.layers import Swish as SwishMe

class SRNet_layer1(nn.Module):

    def __init__(self, in_channels, out_channels, activation=SwishMe(), norm_layer=nn.BatchNorm2d, norm_kwargs={}):
        super(SRNet_layer1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation
        self.norm_layer = norm_layer
        self.conv = ConvolutionalFilterManifold(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3, padding='same', bias=False)#create_conv2d(self.in_channels, self.out_channels, kernel_size=3, stride=1, dilation=1, padding='')
        self.norm = norm_layer(self.out_channels, **norm_kwargs)

    def forward(self, inputs, si):
        x = self.conv(si, inputs)
        x = self.norm(x)
        x = self.activation(x)
        return x


class SRNet_layer2(nn.Module):
    def __init__(self, in_channels, out_channels, activation=SwishMe(), norm_layer=nn.BatchNorm2d, norm_kwargs={}):
        super(SRNet_layer2, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation
        self.norm_layer = norm_layer

        self.layer1 = SRNet_layer1(self.in_channels, self.out_channels, self.activation,
                                  norm_layer=self.norm_layer, norm_kwargs=norm_kwargs)

        self.conv = ConvolutionalFilterManifold(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=3, padding='same',bias=False) 

        self.norm = norm_layer(self.out_channels, **norm_kwargs)

    def forward(self, inputs, si):
        x = self.layer1(inputs, si)
        x = self.conv(si, x)
        x = self.norm(x)
        x = torch.add(x,inputs)
        return x
    
    
class SRNet_layer3(nn.Module):
    def __init__(self, in_channels, out_channels, activation=SwishMe(), norm_layer=nn.BatchNorm2d, norm_kwargs={}):
        super(SRNet_layer3, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation
        self.norm_layer = norm_layer
        
        self.layer1 = SRNet_layer1(self.in_channels, self.out_channels, self.activation,
                                  norm_layer=self.norm_layer, norm_kwargs=norm_kwargs)
        
        self.conv = ConvolutionalFilterManifold(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=3, padding='same', bias=False) 
        
        self.pool = create_pool2d(pool_type='avg', kernel_size=3, stride=2, padding='')
        
        self.norm = norm_layer(self.out_channels, **norm_kwargs)
        
        self.resconv = ConvolutionalFilterManifold(in_channels=self.in_channels, out_channels=self.out_channels, stride=2, kernel_size=1, bias=False) 
        
        self.resnorm = norm_layer(self.out_channels, **norm_kwargs)
        
    def forward(self, inputs, si):
        x = self.layer1(inputs, si)
        x = self.conv(si, x)
        x = self.norm(x)
        x = self.pool(x)
        res = self.resconv(si, inputs)
        res = self.resnorm(res)
        x = torch.add(res,x)
        return x
    
    
class SRNet_layer4(nn.Module):
    def __init__(self, in_channels, out_channels, activation=SwishMe(), norm_layer=nn.BatchNorm2d, norm_kwargs={}):
        super(SRNet_layer4, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation
        self.norm_layer = norm_layer
        
        self.layer1 = SRNet_layer1(self.in_channels, self.out_channels, self.activation,
                                  norm_layer=self.norm_layer, norm_kwargs=norm_kwargs)
        
        self.conv = ConvolutionalFilterManifold(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=3, padding='same', bias=False) 
        
        self.norm = norm_layer(self.out_channels, **norm_kwargs)
        
    def forward(self, inputs, si):
        x = self.layer1(inputs, si)
        x = self.conv(si, x)
        x = self.norm(x)
        return x
    
class TwoInputsSequential(nn.Sequential):
    def forward(self, x, si):
        for module in self._modules.values():
            x = module(x, si)
        return x 


class ManifoldSRNet(nn.Module):
    def __init__(self, in_chans, num_classes, global_pooling='avg', activation=nn.ReLU(), norm_layer=nn.BatchNorm2d, norm_kwargs={}, **kwargs):
        super(ManifoldSRNet, self).__init__()
        self.in_chans = in_chans
        self.activation = activation
        self.norm_layer = norm_layer
        self.num_classes = num_classes
        self.global_pooling = SelectAdaptivePool2d(pool_type=global_pooling, flatten=False)
        
        self.layer_1_specs = [64, 16]
        self.layer_2_specs = [16, 16, 16, 16, 16]
        self.layer_3_specs = [16, 64, 128, 256]
        self.layer_4_specs = [512]
        
        in_channels = self.in_chans
        block1 = []
        for out_channels in self.layer_1_specs:
            block1.append(SRNet_layer1(in_channels, out_channels, activation=self.activation, norm_layer=self.norm_layer, norm_kwargs=norm_kwargs))
            in_channels = out_channels
            
        block2 = []
        for out_channels in self.layer_2_specs:
            block2.append(SRNet_layer2(in_channels, out_channels, activation=self.activation, norm_layer=self.norm_layer, norm_kwargs=norm_kwargs))
            in_channels = out_channels
            
        block3 = []
        for out_channels in self.layer_3_specs:
            block3.append(SRNet_layer3(in_channels, out_channels, activation=self.activation, norm_layer=self.norm_layer, norm_kwargs=norm_kwargs))
            in_channels = out_channels
            
        block4 = []
        for out_channels in self.layer_4_specs:
            block4.append(SRNet_layer4(in_channels, out_channels, activation=self.activation, norm_layer=self.norm_layer, norm_kwargs=norm_kwargs))
            in_channels = out_channels
        
        self.block1 = TwoInputsSequential(*block1)
        self.block2 = TwoInputsSequential(*block2)
        self.block3 = TwoInputsSequential(*block3)
        self.block4 = TwoInputsSequential(*block4)
        
        self.fc = ConvolutionalFilterManifold(in_channels=in_channels, out_channels=self.num_classes, kernel_size=1, padding='same')
        
    def forward_features(self, x, si):
        x = self.block1(x, si)
        x = self.block2(x, si)
        x = self.block3(x, si)
        x = self.block4(x, si)
        return x
    
    def forward(self, x, si):
        x = self.forward_features(x, si)
        x = self.global_pooling(x)
        x = self.fc(si, x)
        return x.squeeze()