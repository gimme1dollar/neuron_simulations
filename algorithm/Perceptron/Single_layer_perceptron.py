import copy
import math
import argparse
import numpy as np

from utils.general_math import sign, cos

class SingleLayerPerceptron():
    def __init__(self, args, **kwargs):
        # print(kwargs['test'])
        self.input_dim = input_dim = 2
        self.output_dim = output_dim = 1

        self.activation_type = activation_type = args.activation_type
        if activation_type == 'sign':
            self.W = np.zeros(shape=(output_dim, input_dim))
            self.b = np.random.rand(output_dim) 

            self.activation_func = sign
        elif activation_type == 'cos':
            self.W = np.array([1, 0]) @ np.array([[1/math.sqrt(2), -1/math.sqrt(2)], [1/math.sqrt(2), 1/math.sqrt(2)]])
            self.b = np.array([0.0]) 
            
            self.activation_func = cos
        self.b_init = copy.deepcopy(self.b)

        self.epoch = args.epoch
        self.learning_rate = args.learning_rate
       
    def train(self, data, label):
        print(self.W, self.b)
        for _ in range(self.epoch):
            for d, l in zip(data, label):
                pred = self.eval(d)

                if pred != l:
                    self.W += self.learning_rate * l * d
                    self.b += self.learning_rate * l * self.b_init 
        print(self.W, self.b)
        return

    def eval(self, data):
        assert data.shape[0] == self.input_dim
        
        pred = np.matmul(self.W, data)
        pred = pred - self.b
        pred = self.activation_func(pred)
        
        return pred

def main(args, **kwargs):
    model = SingleLayerPerceptron(args, **kwargs)
   
    data_name = args.data_name
    if data_name == 'and':
        print('AND problem\n')
        """
          |  
          0  1 
          |  
        --0--0---
        """
        data, label = [(0,0), (0,1), (1,0), (1,1)], \
                        [-1, -1, -1, 1]
        data, label = np.array(data), np.array(label)
    elif data_name == 'xor':
        print('XOR problem\n')
        """
          |  
          1  0
          |  
        --0--1---
        """
        data, label = [(0,0), (0,1), (1,0), (1,1)], \
                        [-1, 1, 1, -1]
        data, label = np.array(data), np.array(label)

    model.train(data, label)

    pred = []
    for d in data:
        p = model.eval(d)
        pred.append(p)
    print(pred)
    return

if __name__ == '__main__':
    from utils.arguments import get_args
    main(get_args())
