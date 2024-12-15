import random
import numpy as np
from neural_network import activation, loss, optimizer

def he_initialisation(input_size: int, size: int) -> np.array:
    return np.random.rand(input_size, size) * np.sqrt(2. / input_size)

class Layer:
    def __init__(self):
        pass

    def forward(self, input, training):
        pass

    def backward(self, output_error):
        pass

class FullyConnectedLayer(Layer):
    def __init__(self, input_size: int, size: int):
        self.weights = he_initialisation(input_size, size)
        self.biases = np.zeros((1, size))

    def forward(self, input, training):
        self.input = input
        self.output = np.dot(self.input, self.weights) + self.biases
        return self.output

    def backward(self, output_error):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)

        self.weights -= 0.01 * weights_error
        self.biases -= 0.01 * np.sum(output_error, axis=0, keepdims=True)
        return input_error

class DropoutLayer(Layer):
    def __init__(self, rate: float):
        self.rate = rate
        self.filter = None

    def forward(self, input, training):
        if training:
            self.filter = np.random.rand(*input.shape) > self.rate
            return (input * self.filter) / (1 - self.rate)
        return input

    def backward(self, output_error):
        if self.filter is None:
            return output_error
        return output_error * self.filter

class ActivationLayer(Layer):
    def __init__(self, activation: callable, activation_prime: callable):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input, training):
        self.input = input
        self.output = self.activation(input)
        return self.output

    def backward(self, output_error):
        return self.activation_prime(self.input) * output_error
