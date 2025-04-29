import random
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from functools import partial

from tools.jpeg_utils import *

def get_train_transforms(type):
    if type.lower() == 'spatial_d4':
        return A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=1.0),
                #A.Resize(height=256, width=256, p=1.0),
                ToTensorV2(p=1.0),
            ], p=1.0, additional_targets={'image2':'image'})
    elif type.lower() == 'jpeg_onehot_d4':
        return jpeg_onehot_d4

def get_valid_transforms(type):
    if type.lower() == 'spatial_d4':
        return A.Compose([
                #A.Resize(height=256, width=256, p=1.0),
                ToTensorV2(p=1.0),
            ], p=1.0, additional_targets={'image2':'image'})
    elif type.lower() == 'jpeg_onehot_d4':
        return partial(jpeg_onehot_d4, rot=0, flip=0)

def jpeg_onehot_d4(image, image2=None, rot=None, flip=None):
    if not rot:
        rot = random.randint(0,3)
    if not flip:
        flip = random.random() < 0.5
    out = {}
    out['image'] = jpeg_abs_bounded_onehot(rot_and_flip_jpeg(image, rot, flip=flip))
    if image2:
        out['image2'] = jpeg_abs_bounded_onehot(rot_and_flip_jpeg(image2, rot, flip=flip))
    return out