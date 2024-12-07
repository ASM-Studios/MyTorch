from neural_network import layer, activation
import numpy as np
import random
import time
import pickle

class NeuralNetwork:
    def __init__(self) -> None:
        self.__layers = []
        self.__loss = None
        self.__loss_prime = None

    def __forward(self, input: list):
        output = input
        for layer in self.__layers:
            output = layer.forward(output)
        return output

    def get_layers(self):
        return self.__layers

    def add_layer(self, layer: layer.Layer):
        self.__layers.append(layer)

    def set_loss_functions(self, loss: callable, loss_prime: callable) -> None:
        self.__loss = loss
        self.__loss_prime = loss_prime

    def train(self, x_train, y_train, epochs: int, batch_size: int = 16):
        if (self.__loss == None or self.__loss_prime == None):
            raise NotImplementedError

        x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
        y_train = np.array([[[0]], [[1]], [[1]], [[0]]])
        for _ in range(epochs):
            index = random.randint(0, 3)
            output = x_train[index]
            
            output = self.__forward(output)
            error = self.__loss_prime(y_train[index], output)

            for i, layer in enumerate(reversed(self.__layers)):
                error = layer.backward(error, 0.1)
        return 0

    def predict(self, input: set):
        inputs = input
        return self.__forward(inputs)

    def save(self, filename: str):
        pickle.dump(self, open(filename, 'wb'))
        return 0

    def restore(self, filename: str):
        return 0
