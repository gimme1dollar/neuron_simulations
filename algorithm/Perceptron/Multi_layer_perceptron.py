import copy
import math
import argparse
import numpy as np

class MultiLayerPerceptron():
    def __init__(self, args, **kwargs):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation_func = sigmoid
        
        self.W = np.zeros(shape=(output_dim, input_dim))
        self.b = np.random.rand(output_dim) 

        self.epoch = epoch
        self.learning_rate = learning_rate
       
    def train(self, data, label):
        return

    def eval(self, data):
        pred = None
        return pred

def main(args):
    return
