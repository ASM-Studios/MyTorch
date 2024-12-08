from neural_network import layer, activation, neural_network
from neural_network.loss import loss_functions
from neural_network.activation import activation_functions
import numpy as np
import random
import time
import pickle
import tqdm

class NeuralNetwork:
    def __init__(self) -> None:
        self.layers = []
        self.loss = None
        self.loss_prime = None

    def __forward(self, input: list):
        output = input
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def add_layer(self, layer: layer.Layer):
        self.layers.append(layer)

    def set_loss_functions(self, loss: callable, loss_prime: callable) -> None:
        self.__loss = loss
        self.__loss_prime = loss_prime

    """def __train_batch(self, x_batch, y_batch, learning_rate):
        output = self.__forward(x_batch)
        error = self.__loss_prime(y_batch, output)
        for i, layer in enumerate(reversed(self.__layers)):
            error = layer.backward(error, learning_rate)"""

    def train(self, x_train, y_train, epochs: int, batch_size: int = 16):
        if (self.__loss == None or self.__loss_prime == None):
            raise NotImplementedError

        for epoch in tqdm.tqdm(range(epochs)):
            loss = []
            for i in range(0, len(x_train), 1):
                """x_batch = x_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]
                self.__train_batch(x_batch, y_batch, 0.1)
                continue"""
                output = self.__forward(x_train[i])
                loss.append(self.__loss(y_train[i], output))

                error = self.__loss_prime(y_train[i], output)
                for layer in reversed(self.layers):
                    error = layer.backward(error)
            loss = np.mean(loss)
            tqdm.tqdm.write(f'Epoch: {epoch}, Loss: {loss}')



        return 0

    def predict(self, input: set):
        return self.__forward(input)

    def save(self, filename: str):
        pickle.dump(self, open(filename, 'wb'))
        return

    @staticmethod
    def restore(filename: str):
        return pickle.load(open(filename, 'rb'))

