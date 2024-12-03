#!/usr/bin/env python3

import sys
import random

from neural_network import neural_network, activation
import numpy as np

def MSE_loss(results: list):
    loss = 0
    N = len(results)
    if (N == 0):
        return 0
    for result, predict in results:
        loss += (result - predict) ** 2
    return loss / N

def cross_entropy_loss(results: list):
    loss = 0
    N = len(results)
    if (N == 0):
        return 0
    epsilon = 1e-10
    for result, predict in results:
        if predict == 0:
            predict += epsilon
        elif predict == 1:
            predict -= epsilon
        loss += result * np.log(predict) + (1 - result) * np.log(1 - predict)
    return -loss / N

def generate_train():
    input_l = random.randint(0, 1)
    input_r = random.randint(0, 1)
    output = input_l ^ input_r
    return ((input_l, input_r), [output])

if __name__ == '__main__':
    datas = [generate_train() for i in range(1000)]
    nn = neural_network.NeuralNetwork(2, 0, 1, cross_entropy_loss)
    nn.train(datas, 3)
    #print(nn.predict((0, 0)))
    #print(nn.predict((1, 0)))
    #print(nn.predict((0, 1)))
    #print(nn.predict((1, 1)))
    sys.exit(0)
