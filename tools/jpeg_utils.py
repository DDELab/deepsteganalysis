import numpy as np
import jpegio as jio
from tqdm import tqdm
import os
from scipy import fftpack
from numpy.lib.stride_tricks import as_strided
from collections import defaultdict 
import torch

quantization_dict = dict()
quantization_dict[95] = np.array([[ 2,  1,  1,  2,  2,  4,  5,  6],
                                  [ 1,  1,  1,  2,  3,  6,  6,  6],
                                  [ 1,  1,  2,  2,  4,  6,  7,  6],
                                  [ 1,  2,  2,  3,  5,  9,  8,  6],
                                  [ 2,  2,  4,  6,  7, 11, 10,  8],
                                  [ 2,  4,  6,  6,  8, 10, 11,  9],
                                  [ 5,  6,  8,  9, 10, 12, 12, 10],
                                  [ 7,  9, 10, 10, 11, 10, 10, 10]])
quantization_dict[75] = np.array([[ 8,  6,  5,  8, 12, 20, 26, 31],
                                  [ 6,  6,  7, 10, 13, 29, 30, 28],
                                  [ 7,  7,  8, 12, 20, 29, 35, 28],
                                  [ 7,  9, 11, 15, 26, 44, 40, 31],
                                  [ 9, 11, 19, 28, 34, 55, 52, 39],
                                  [12, 18, 28, 32, 41, 52, 57, 46],
                                  [25, 32, 39, 44, 52, 61, 60, 51],
                                  [36, 46, 48, 49, 56, 50, 52, 50]])
quantization_dict[90] = np.array([[ 3,  2,  2,  3,  5,  8, 10, 12],
                                  [ 2,  2,  3,  4,  5, 12, 12, 11],
                                  [ 3,  3,  3,  5,  8, 11, 14, 11],
                                  [ 3,  3,  4,  6, 10, 17, 16, 12],
                                  [ 4,  4,  7, 11, 14, 22, 21, 15],
                                  [ 5,  7, 11, 13, 16, 21, 23, 18],
                                  [10, 13, 16, 17, 21, 24, 24, 20]
                                  ,[14, 18, 19, 20, 22, 20, 21, 20]])

def ycbcr2rgb(im):
    xform = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
    rgb = im.astype(np.float32)
    rgb[:,:,[1,2]] -= 128
    rgb = rgb.dot(xform.T)
    return rgb

def block_view(A, block= (8,8)):
    """Provide a 2D block view to 2D array. No error checking made.
    Therefore meaningful (as implemented) only for blocks strictly
    compatible with the shape of A."""
    # simple shape and strides computations may seem at first strange
    # unless one is able to recognize the 'tuple additions' involved ;-)
    shape= (A.shape[0]// block[0], A.shape[1]// block[1])+ block
    strides= (block[0]* A.strides[0], block[1]* A.strides[1])+ A.strides
    return as_strided(A, shape= shape, strides= strides)

def segmented_stride(M, fun, blk_size=(8,8), overlap=(0,0)):
    # This is some complex function of blk_size and M.shape
    B = block_view(M, block=blk_size)
    B[:,:,:,:] = fun(B)
    return M

def decompress_structure(S):
    # Decompress DCT coefficients C using quantization table Q
    H = S.coef_arrays[0].shape[0]
    W = S.coef_arrays[0].shape[1]
    n = len(S.coef_arrays)
    assert H % 8 == 0, 'Wrong image size'
    assert W % 8 == 0, 'Wrong image size'
    I = np.zeros((H,W,n),dtype=np.float64) # Returns Y, Cb and Cr
    for i in range(n):
        Q = S.quant_tables[S.comp_info[i].quant_tbl_no]
        # this multiplication is done on integers
        fun = lambda x : np.multiply(x,Q)
        C = np.float64(segmented_stride(S.coef_arrays[i], fun)) 
        fun = lambda x: fftpack.idct(fftpack.idct(x, norm='ortho',axis=2), norm='ortho',axis=3) + 128
        I[:,:,i] = segmented_stride(C, fun)
    return I


def rot_and_flip_jpeg(S, rot, flip=False):
    """Lossless rotation and flips in the DCT domain"""
    assert S.coef_arrays[0].shape[0] % 8 == 0, 'Image height is not a multiple of 8!'
    assert S.coef_arrays[0].shape[1] % 8 == 0, 'Image width is not a multiple of 8!'
    I = np.empty([S.coef_arrays[0].shape[0], S.coef_arrays[0].shape[1], S.image_components],dtype=np.int32)
    for ch in range(I.shape[-1]):
        C = np.copy(S.coef_arrays[ch])  
        C = np.rot90(C, k=rot)
        fun = lambda x: np.rot90(x,k=4-rot, axes=(2,3))
        C = segmented_stride(C, fun)
        # transpose
        if rot in [1,3]:
            fun = lambda x: np.copy(np.transpose(x, axes=[0,1,3,2]))
            C = segmented_stride(C, fun)
            for q in range(len(S.quant_tables)):
                S.quant_tables[q] = S.quant_tables[q].T
        # multiply even rows by -1
        if rot in [1,2]:
            C[1::2,:] *= -1
        # multiply even columns by -1
        if rot in [2,3]:
            C[:,1::2] *= -1  
        # flip
        if flip == True:
            C = np.fliplr(C)
            fun = lambda x: np.flip(x, axis=3)
            C = segmented_stride(C, fun)
            C[:,1::2] *= -1 
        S.coef_arrays[ch] = C[:,:]
    return S

def jpeg_abs_bounded_onehot(tmp,T=5):
    """Threshold + abs + onehot encoding"""
    I = np.dstack(tmp.coef_arrays)
    I = np.abs(I)
    I = np.clip(I, a_min=0,a_max=T)
    I_oh = np.zeros((I.shape)+(T+1,))
    for ch in range(I.shape[-1]):
        I_oh[:,:,ch,:] = (I[:,:,ch,np.newaxis] == range(T+1))
    I_oh = I_oh.reshape((I.shape[0],I.shape[1], -1)).astype(np.float32)
    return torch.from_numpy(I_oh.transpose(2,0,1)) 