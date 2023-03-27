import math
import numpy as np

def sign(x):
    if x >= 0: return 1
    else: return -1

def cos(x):
    out = np.cos(x)
    if out >= 0.8: return 1
    else: return -1

