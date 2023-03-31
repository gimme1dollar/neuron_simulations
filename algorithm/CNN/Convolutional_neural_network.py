import copy
import math
import argparse
import numpy as np

from utils.general_math import relu, relu_grad

class ConvolutionalNeuralNetwork():
    def __init__(self, args, **kwargs):
        self.input_dim = args.input_dim
        self.hidden_dim = args.hidden_dim
        self.output_dim = args.output_dim
        self.activation_func = relu
        
        self.kernel = None

        self.epoch = args.epoch
        self.learning_rate = args.learning_rate
       
    def train(self, data, label):
        # forward

        # loss

        # backward

        # optimize step
        return

    def eval(self, data):
        '''
        '''
        return 

def main(args):
    return

