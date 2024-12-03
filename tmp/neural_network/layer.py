import random
import numpy as np

def nothing(input: float) -> float:
    return input

class Layer:
    def __init__(self, input_size: int, size: int, activation: callable = nothing):
        self.__weight = np.random.rand(size, input_size)
        self.__bias = np.random.rand(size)
        self.__input_size = input_size
        self.__size = size
        self.__activation = activation

    def get_input_size(self):
        return self.__input_size

    def get_size(self):
        return self.__size

    def get_weights(self):
        return self.__weight

    def forward(self, input_args: list):
        res = []
        for i, neuron in enumerate(self.__weight):
            res.append(self.__activation(np.dot(neuron, input_args) + self.__bias[i]))
        return res

    def backward(self, output: list, lr: float):
        print(output)
        pass

class InputLayer(Layer):
    def __init__(self, size: int):
        super().__init__(size, size, nothing)

    def forward(self, input: list):
        return input

class OutputLayer(Layer):
    def backward(self, input: list, lr: float):
        for i in range(0, len(self.__weight)):
            predict, result = input[i]
            oue = predict * (1 - predict) * (result - predict)
            for j in range(0, len(self.__weight[i])):
                delta = lr * oue * self.__weight[i][j]
                self.__weight[i][j] += delta
