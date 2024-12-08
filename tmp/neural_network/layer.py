import random
import numpy as np

def nothing(input: float) -> float:
    return input

class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        raise NotImplementedError

    def backward(self, output_error, lr):
        raise NotImplementedError

class FCLayer(Layer):
    def __init__(self, input_size: int, size: int):
        self.weights = np.random.rand(input_size, size)
        self.biases = np.random.rand(1, size)

    def forward(self, input):
        self.input = input
        self.output = np.dot(self.input, self.weights) + self.biases
        return self.output

    def backward(self, output_error, lr):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)

        self.weights -= lr * weights_error
        self.biases -= lr * output_error
        return input_error

class ActivationLayer(Layer):
    def __init__(self, activation: callable, activation_prime: callable):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        self.input = input
        self.output = self.activation(input)
        return self.output

    def backward(self, output_error, lr):
        return self.activation_prime(self.input) * output_error
