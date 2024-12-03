from neural_network import layer, activation
import numpy as np
import random

class NeuralNetwork:
    def __init__(self, input: int, hidden: int, output: int, loss: callable):
        self.__input = layer.InputLayer(2)
        self.__hidden = []
        self.__output = layer.OutputLayer(2, 1, activation.heaviside_function)
        self.__lr = 0.1
        self.__loss = loss

    def __forward(self, input: list):
        res = self.__input.forward(input)
        for i, layer in enumerate(self.__hidden):
            res = layer.forward(res)
        res = self.__output.forward(res)
        return res

    def train(self, input: list, epochs: int, batch_size: int = 16):
        for _ in range(epochs):
            results = []
            for j in range(batch_size):
                inputs, output = random.choice(input)
                predict = self.__forward(inputs)
                results.append((predict, output))
            #loss = self.__loss(results)
            print('Result:\n' + str(results))
            self.__output.backward(results, self.__lr)
            #print(loss)
        return 0

    def predict(self, input: set):
        inputs = input
        return self.__forward(inputs)

    def save(self, filename: str):
        return 0

    def restore(self, filename: str):
        return 0
