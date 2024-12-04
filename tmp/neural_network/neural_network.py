from neural_network import layer, activation
import numpy as np
import random
import tqdm
import time

class NeuralNetwork:
    def __init__(self, loss: callable, loss_prime: callable) -> None:
        self.__layers = []
        self.__loss = loss
        self.__loss_prime = loss_prime

    def add(self, layer: layer.Layer):
        self.__layers.append(layer)

    def __forward(self, input: list):
        output = input
        for layer in self.__layers:
            output = layer.forward(output)
        return output

    def train(self, inputs: list, epochs: int, batch_size: int = 16):
        for _ in tqdm.tqdm(range(epochs)):
            input, expected_output = random.choice(inputs)
            predict = self.__forward(input)
            error = self.__loss_prime(expected_output, predict)
            print(error)
            for i, layer in enumerate(reversed(self.__layers)):
                error = layer.backward(error, 0.01)
        return 0

    def predict(self, input: set):
        inputs = input
        return self.__forward(inputs)

    def save(self, filename: str):
        return 0

    def restore(self, filename: str):
        return 0
