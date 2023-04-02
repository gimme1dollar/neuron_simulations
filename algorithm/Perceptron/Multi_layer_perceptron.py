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
        self.b1 = np.zeros(shape=(self.hidden_dim, 1)) 
        self.W2 = np.random.normal(0, 1, size=(self.output_dim, self.hidden_dim))
        self.b2 = np.zeros(shape=(self.output_dim, 1))

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
        dLdW2 = np.dot(dLda2, feat.transpose()) / len(data) 
        dLdb2 = np.sum(dLda2) / len(data)

        dLda1 = sigmoid_grad(feat) * np.dot(self.W2.transpose(), dLda2)
        dLdW1 = np.dot(dLda1, data.transpose()) / len(data)
        dLdb1 = np.sum(dLda1) / len(data)

        # optimize step
        self.W2 -= self.learning_rate * dLdW2
        self.b2 -= self.learning_rate * dLdb2
        self.W1 -= self.learning_rate * dLdW1
        self.b1 -= self.learning_rate * dLdb1

        return L

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

def build_batches(train_data, train_labels, batch_size=512):
    batches_num = len(train_data) // batch_size
    if len(train_data) % batch_size != 0: batches_num += 1
    
    batches_data, batches_label = [], []
    for i in range(batches_num):
        start, end   = i*batch_size, (i+1)*batch_size
        batch_data   = train_data  [start:end]
        batch_labels = train_labels[start:end]

        batches_data.append(batch_data)
        batches_label.append(batch_labels)
    return batches_data, batches_label

def main(args, **kwargs):

    # model
    model = MultiLayerPerceptron(args, **kwargs)

    # dataset
    from data.MNIST import get_data
    train_data, train_labels, \
            test_data, test_labels = get_data()

    # training
    losses = []
    for e in range(args.epoch):
        batches_data, batches_label = build_batches(train_data, train_labels, args.batch_size)

        loss = 0
        for batch_data, batch_label in zip(batches_data, batches_label):
            loss += model.train(batch_data.transpose(), batch_label.transpose())
        
        losses.append(loss)

        if e % 10 == 0:
            print('training ', e, ' :', loss)
    print()
        
    # evaluation
    batches_data, batches_label = build_batches(test_data, test_labels, args.batch_size)

    loss = 0
    for batch_data, batch_label in zip(batches_data, batches_label):
        pred, _ = model.eval(batch_data.transpose())

        L = np.sum(batch_label.transpose() * np.log(pred)) # cross-entropy loss
        L = L / len(batch_data)
        L = -1. * L

        loss += L
    print('evaluation :', loss)

    return

