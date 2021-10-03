dependencies = ['torch']

from models.models import get_net

def jin_srnet(pretrained=False, progress=True,**kwargs):
    return get_net('srnet')