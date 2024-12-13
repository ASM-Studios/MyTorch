import random
import numpy as np
from neural_network import activation, loss, optimizer

class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        raise NotImplementedError

    def backward(self, output_error):
        raise NotImplementedError

class FCLayer(Layer):
    def __init__(self, input_size: int, size: int):
        self.weights = np.random.rand(input_size, size) * np.sqrt(2. / input_size)
        self.biases = np.zeros((1, size))
        self.optimizer = optimizer.SGDOptimizer(0.1)

    def forward(self, input):
        self.input = input
        self.output = np.dot(self.input, self.weights) + self.biases
        return self.output

    def backward(self, output_error):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)

        #self.weights = self.optimizer.update(self.weights, weights_error)
        #self.biases = self.optimizer.update(self.biases, output_error)
        self.weights -= 0.01 * weights_error
        self.biases -= 0.01 * np.sum(output_error, axis=0, keepdims=True)
        return input_error

class ActivationLayer(Layer):
    def __init__(self, activation: callable, activation_prime: callable):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        self.input = input
        self.output = self.activation(input)
        return self.output

    def backward(self, output_error):
        return self.activation_prime(self.input) * output_error
