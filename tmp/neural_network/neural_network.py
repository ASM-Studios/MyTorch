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
        x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
        y_train = np.array([[[0]], [[1]], [[1]], [[0]]])
        for _ in tqdm.tqdm(range(epochs)):
            index = random.randint(0, 3)
            output = x_train[index]
            
            output = self.__forward(output)
            error = self.__loss_prime(y_train[index], output)
            
            print(len(error))
            print(len(self.__layers))
            for i, layer in enumerate(reversed(self.__layers)):
                error = layer.backward(error, 0.1)
                print('>', len(error), error)
        return 0

    def predict(self, input: set):
        inputs = input
        return self.__forward(inputs)

    def save(self, filename: str):
        return 0

    def restore(self, filename: str):
        return 0
