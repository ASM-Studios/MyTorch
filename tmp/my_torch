#!/usr/bin/env python3

import sys
import random

from neural_network import neural_network, activation, layer
import numpy as np

def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2));

def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / 1;

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
    return (np.array([input_l, input_r]), np.array([output]))

if __name__ == '__main__':
    input = np.array([0, 1, 2])
    output = np.array(1)

    print(np.dot(input, output))

    datas = [generate_train() for i in range(1000)]
    nn = neural_network.NeuralNetwork(mse, mse_prime)
    nn.add(neural_network.layer.FCLayer(2, 3))
    nn.add(neural_network.layer.ActivationLayer(activation.tanh, activation.tanh_prime))
    nn.add(neural_network.layer.FCLayer(3, 1))
    nn.add(neural_network.layer.ActivationLayer(activation.tanh, activation.tanh_prime))

    nn.train(datas, 1000)
    print(nn.predict([[1, 1]]))
    print(nn.predict([[0, 0]]))
    print(nn.predict([[0, 1]]))
    print(nn.predict([[1, 0]]))
    sys.exit(0)
