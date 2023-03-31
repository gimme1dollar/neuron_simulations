import copy
import math
import argparse
import numpy as np

from utils.general_math import sigmoid, sigmoid_grad, softmax

class MultiLayerPerceptron():
    def __init__(self, args, **kwargs):
        self.input_dim = args.input_dim
        self.hidden_dim = args.hidden_dim
        self.output_dim = args.output_dim
        self.activation_func = sigmoid
        
        self.W1 = np.random.normal(0, 1, size=(self.hidden_dim, self.input_dim))
        self.b1 = np.zeros(self.hidden_dim) 
        self.W2 = np.random.normal(0, 1, size=(self.output_dim, self.hidden_dim))
        self.b2 = np.zeros(self.output_dim)

        self.epoch = args.epoch
        self.learning_rate = args.learning_rate
       
    def train(self, data, label):
        # forward
        pred, feat = self.eval(data)

        # loss
        L = np.sum(label * np.log(pred)) # cross-entropy loss
        L = L / len(data)
        L = -1. * L

        # backward
        dLda2 = pred - label # CE = log( sum_j (a2_j) ) - a2_l
        dLdW2 = np.dot(np.transpose(dLda2), feat) / len(data) 
        dLdb2 = np.sum(dLda2) / len(data)

        dLda1 = sigmoid_grad(feat) * np.dot(dLda2, self.W2)
        dLdW1 = np.dot(dLda1, data) / len(data)
        dLdb1 = np.sum(dLda1) / len(data)

        # optimize step
        self.W2 -= self.learning_rate * dLdW2
        self.b2 -= self.learning_rate * dLdb2
        self.W1 -= self.learning_rate * dLdW1
        self.b1 -= self.learning_rate * dLdb1

        return

    def eval(self, data):
        '''
            (x) -> [W1] -> (a1) -> sigmoid -> (h)* -> [W2] -> (a2) -> softmax -> (y)
                   [b1] /                             [b2] /
        '''
        a1 = np.dot(self.W1, data) + self.b1
        h = self.activation_func(a1)
        a2 = np.dot(self.W2, h) + self.b2
        y = softmax(a2)

        return y, h

def main(args):
    return

