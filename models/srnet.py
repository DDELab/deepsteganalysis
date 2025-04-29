'''
"Deep Residual Network for Steganalysis of Digital Images"
Mehdi Boroumand, Mo Chen, and Jessica Fridrich
http://www.ws.binghamton.edu/fridrich/Research/SRNet.pdf
'''
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# from timm.models.layers.activations_me import SwishMe
from timm.models.layers import Swish as SwishMe
import torch
from timm.models.layers import create_conv2d, create_pool2d
from torch import nn

class SRNet_layer1(nn.Module):

    def __init__(self, in_channels, out_channels, activation=SwishMe(), norm_layer=nn.BatchNorm2d, norm_kwargs={}):
        super(SRNet_layer1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation
        self.norm_layer = norm_layer
        self.conv = create_conv2d(self.in_channels, self.out_channels, kernel_size=3, stride=1, dilation=1, padding='')
        self.norm = norm_layer(self.out_channels, **norm_kwargs)

    def forward(self, inputs):
        x = self.conv(inputs)
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

        self.conv = create_conv2d(self.out_channels, self.out_channels,
                                  kernel_size=3, stride=1, dilation=1, padding='')

        self.norm = norm_layer(self.out_channels, **norm_kwargs)

    def forward(self, inputs):
        x = self.layer1(inputs)
        x = self.conv(x)
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
        
        self.conv = create_conv2d(self.out_channels, self.out_channels, 
                                  kernel_size=3, stride=1, dilation=1, padding='')
        
        self.pool = create_pool2d(pool_type='avg', kernel_size=3, stride=2, padding='')
        
        self.norm = norm_layer(self.out_channels, **norm_kwargs)
        
        self.resconv = create_conv2d(self.in_channels, self.out_channels, 
                                  kernel_size=1, stride=2, dilation=1, padding='')
        
        self.resnorm = norm_layer(self.out_channels, **norm_kwargs)
        
    def forward(self, inputs):
        x = self.layer1(inputs)
        x = self.conv(x)
        x = self.norm(x)
        x = self.pool(x)
        res = self.resconv(inputs)
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
        
        self.conv = create_conv2d(self.out_channels, self.out_channels, 
                                  kernel_size=3, stride=1, dilation=1, padding='')
        
        self.norm = norm_layer(self.out_channels, **norm_kwargs)
        
    def forward(self, inputs):
        x = self.layer1(inputs)
        x = self.conv(x)
        x = self.norm(x)
        return x
    
    
class SRNet(nn.Module):
    def __init__(self, in_chans, num_classes, activation=nn.ReLU(), norm_layer=nn.BatchNorm2d, norm_kwargs={}, **kwargs):
        super(SRNet, self).__init__()
        self.in_chans = in_chans
        self.activation = activation
        self.norm_layer = norm_layer
        self.num_classes = num_classes
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        
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
        
        self.block1 = nn.Sequential(*block1)
        self.block2 = nn.Sequential(*block2)
        self.block3 = nn.Sequential(*block3)
        self.block4 = nn.Sequential(*block4)
        
        self.fc = nn.Linear(in_channels, self.num_classes, bias=True)
        
    def forward_features(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x
    
    def forward(self, x):
        x = self.forward_features(x)
        x = self.global_pooling(x).squeeze((2,3))
        x = self.fc(x)
        return x