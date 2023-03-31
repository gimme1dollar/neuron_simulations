import math
import numpy as np

def sign(x):
    if x >= 0: return 1
    else: return -1

def cos(x):
    out = np.cos(x)
    if out >= 0.8: return 1
    else: return -1

def sigmoid(x):
    return 1. / (1. + np.exp(-1 * x))

def sigmoid_grad(x):
    return x * (1 - x)

def softmax(x):
    res = np.exp(x) 
    res = res / np.sum(res, axis=1, keepdims=True) # x \in [output_dim, class_num]
    return res

def relu(x):
    if x < 0: return 0
    else: return x
