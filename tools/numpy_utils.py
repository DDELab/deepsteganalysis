import numpy as np

def check_nans(*args):
    for ar in args:
        if np.isnan(ar).any():
            return True
    return False