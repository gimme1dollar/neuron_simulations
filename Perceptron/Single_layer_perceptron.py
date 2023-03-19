import copy
import argparse
import numpy as np

def sign(x):
    if x >= 0: return 1
    else: return -1

def sin(x):
    sinx = np.sin(x)
    if sinx >= 0.5: return 1
    else: return -1

class SingleLayerPerceptron():
    def __init__(self,\
                 input_dim=2, output_dim=1,\
                 activation_type='sign',
                 epoch=100, learning_rate=1e-3
                 ):
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.W = np.zeros(shape=(output_dim, input_dim))
        self.b = np.random.rand(output_dim) 
        self.b_init = copy.deepcopy(self.b)

        if activation_type == 'sign':
            self.activation_func = sign
        elif activation_type == 'sin':
            self.activation_func = sin

        self.epoch = epoch
        self.learning_rate = learning_rate
       
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

        if pred >= 0.5: return 1
        else: return -1

def run_and(args):
    """
      |  
      0  1
      |  
    --0--0---
    """
    data, label = [(0,0), (0,1), (1,0), (1,1)], \
                    [-1, -1, -1, 1]
    data, label = np.array(data), np.array(label)

    model = SingleLayerPerceptron(\
                input_dim=2,\
                output_dim=1,\
                activation_type=args.activation_type,\
                epoch=100,\
                learning_rate=1e-1)
    model.train(data, label)

    pred = []
    for d in data:
        p = model.eval(d)
        pred.append(p)
    print(pred)
    return

def run_xor(args):
    """
      |  
      1  0
      |  
    --0--1---
    """
    data, label = [(0,0), (0,1), (1,0), (1,1)], \
                    [-1, 1, 1, -1]
    data, label = np.array(data), np.array(label)

    model = SingleLayerPerceptron(\
                input_dim=2,\
                output_dim=1,\
                activation_type=args.activation_type,\
                epoch=100,\
                learning_rate=1e-1)
    model.train(data, label)

    pred = []
    for d in data:
        p = model.eval(d)
        pred.append(p)
    print(pred)
    return

def main(args):
    exp_name = args.data_name
    if exp_name == 'and':
        run_and(args)
    elif exp_name == 'xor':
        run_xor(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name')
    parser.add_argument('--activation_type')
    args = parser.parse_args()

    main(args)
