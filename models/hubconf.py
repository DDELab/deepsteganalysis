dependencies = ['torch', 'timm']
from models.models import get_net

# resnet18 is the name of entrypoint
def srnet(num_classes=2, in_chans=3, pretrained=False, **kwargs):
    """ 
    SRNet
    http://www.ws.binghamton.edu/fridrich/Research/SRNet.pdf
    If pretrained this will load JIN pretrained SRNet
    http://www.ws.binghamton.edu/fridrich/Research/IN-pretraining-6.pdf
    """
    if pretrained:
        ckpt_path = 'https://github.com/DDELab/deepsteganalysis/releases/download/v0.1/jin_srnet.ckpt'
    else:
        ckpt_path = None
    model = get_net('srnet', num_classes=num_classes, in_chans=in_chans, ckpt_path=ckpt_path)
    return model