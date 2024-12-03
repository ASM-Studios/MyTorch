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

"""class Layer:
    def __init__(self, input_size: int, size: int, activation: callable, activation_derivative: callable, lr: float):
                self.__grad_weight = np.zeros((size, input_size))
        self.__grad_bias = np.zeros(size)

        self.__input_size = input_size
        self.__size = size
        self.__activation = activation
        self.__activation_derivative = activation_derivative

        self.__lr = lr
        self.__input = []

    def get_input_size(self):
        return self.__input_size

    def get_size(self):
        return self.__size

    def get_weights(self):
        return self.__weight

    def forward(self, input_args: list):
        res = []
        self.__input = input_args
        for i, neuron in enumerate(self.__weight):
            res.append(self.__activation(np.dot(neuron, input_args) + self.__bias[i]))
        return res

    def apply_gradient(self):
        self.__weight += self.__grad_weight * self.__lr
        self.__bias += self.__grad_bias * self.__lr

    def backward(self, output: list, lr: float):
        print(output)
        pass

class OutputLayer(Layer):
    def backward(self, result: set):
        for i in range(0, len(self._Layer__weight)):
            predict, result = result
            error = result - predict
            gradient = error * self._Layer__activation_derivative(predict)
            self._Layer__grad_weight[i] = gradient * self._Layer__lr * self._Layer__input[i]
        self._Layer__grad_bias = gradient * self._Layer__lr"""
