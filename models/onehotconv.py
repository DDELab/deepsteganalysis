'''
"An Intriguing Struggle of CNNs in JPEG Steganalysis and the OneHot Solution"
Yassine Yousfi and Jessica Fridrich
http://www.ws.binghamton.edu/fridrich/Research/OneHot_Revised.pdf
Code adapted from 
https://github.com/YassineYousfi/OneHotConv
'''
from functools import partial
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import torch
from timm.models.layers import create_conv2d, create_pool2d
from torch import nn

from models.srnet import SRNet_layer1

class OneHotConv_layer1(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.ReLU(), norm_layer=nn.BatchNorm2d, norm_kwargs={}):
        super(OneHotConv_layer1, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation
        
        self.conv_dilated = create_conv2d(self.in_channels, self.out_channels//2, 
                                  kernel_size=3, stride=1, dilation=8, padding='')
        
        self.norm_dilated = norm_layer(self.out_channels//2, **norm_kwargs)
        
        self.conv = create_conv2d(self.in_channels, self.out_channels//2, 
                                  kernel_size=3, stride=1, dilation=1, padding='')
        
        self.norm = norm_layer(self.out_channels//2, **norm_kwargs)
        
    def forward(self, inputs):
        x = self.conv_dilated(inputs)
        x = self.norm_dilated(x)
        y = self.conv(inputs)
        y = self.norm(y)
        y = torch.cat((x,y), 1)
        y = self.activation(y)
        return y
    
    
class OneHotConv(nn.Module):
    def __init__(self, in_chans, num_classes, out_channels=32, activation=nn.ReLU(), norm_layer=nn.BatchNorm2d, norm_kwargs={}, pretrained=False):
        super(OneHotConv, self).__init__()
        self.in_chans = in_chans
        self.activation = activation
        self.norm_layer = norm_layer
        self.num_classes = num_classes
        self.out_channels = out_channels

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.out_channels, self.num_classes, bias=True) 
        self.layer1 = OneHotConv_layer1(self.in_chans, 2*self.out_channels, activation=self.activation, norm_layer=self.norm_layer, norm_kwargs=norm_kwargs)
        self.layer2 = SRNet_layer1(2*self.out_channels, self.out_channels, activation=self.activation, norm_layer=self.norm_layer, norm_kwargs=norm_kwargs)
        
    def forward_features(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.global_pooling(x).squeeze((2,3))
        return x
    
    def forward(self, x):
        x = self.forward_features(x)
        x = self.fc(x)
        return x