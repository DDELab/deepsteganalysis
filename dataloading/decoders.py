import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import jpegio as jio
import numpy as np
import binascii
import cv2
from itertools import compress

from torch.utils.data import Dataset, DataLoader
from tools.jpeg_utils import *
from tools.stego_utils import *

def load_or_pass(x, type=jio.decompressedjpeg.DecompressedJpeg, load_fn=jio.read):
    return x if isinstance(x, type) else load_fn(x)

def encode_string(s, max=50):
    h = binascii.hexlify(s.encode('utf-8'))
    result = [int(h[i:i+8], 16) for i in range(0, len(h), 8)]
    return result+[-1]*(max-len(result))

def decode_string(l):
    result = ''
    for s in l:
        if s == -1:
            break
        h = hex(s)[2:].encode('ascii')
        result += binascii.unhexlify(h).decode('utf-8')
    return result

def decoder2in_chans(decoder):
    if decoder == 'ycbcr':
        return 3
    elif decoder == 'rgb':
        return 3
    elif decoder == 'y':
        return 1
    elif decoder == 'rjca':
        return 1
    elif decoder == 'gray_spatial':
        return 1
    elif decoder == 'onehot':
        return 6

def ycbcr_decode(path):
    tmp = load_or_pass(path)
    image = decompress_structure(tmp)
    image = image[:,:,:].astype(np.float32)
    image /= 255.0
    return image

def rgb_decode(path):
    tmp = load_or_pass(path)
    image = decompress_structure(tmp)
    image = image[:,:,:].astype(np.float32)
    image = ycbcr2rgb(image).astype(np.float32)
    image /= 255.0
    return image

def y_decode(path):
    tmp = load_or_pass(path)
    image = decompress_structure(tmp)
    image = image[:,:,:1].astype(np.float32)
    image /= 255.0
    return image

def onehot_decode(path):
    return load_or_pass(path)

def rjca_decode(path):
    tmp = load_or_pass(path)
    image = decompress_structure(tmp)
    image = image[:,:,:1].astype(np.float32)
    return image - np.round(image)

def gray_spatial_decode(path):
    image = load_or_pass(path, np.ndarray, cv2.imread)
    image = image[:,:,:1].astype(np.float32)
    image /= 255.0
    return image

def cost_map_decode(path, cover_path, payload):
    cost_map = np.load(path)
    cover = jio.read(cover_path)
    nzac = np.count_nonzero(cover.coef_arrays[0]) - np.count_nonzero(cover.coef_arrays[0][::8,::8])
    stego = embedding_simulator(cover.coef_arrays[0], cost_map['rho_p1'], cost_map['rho_m1'], nzac*payload)
    cover.coef_arrays[0] = stego
    return cover

def change_map_decode(path, cover_path):
    change_map = np.load(path)
    cover = jio.read(cover_path)
    stego = sample_stego_image(cover.coef_arrays[0], change_map['pChangeP1'], change_map['pChangeM1'])
    cover.coef_arrays[0] = stego
    return cover


