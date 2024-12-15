from neural_network import layer, activation, neural_network
from neural_network.loss import loss_functions
from neural_network.activation import activation_functions
import numpy as np
import random
import time
import pickle
import tqdm
import sys

class NeuralNetwork:
    def __init__(self) -> None:
        self.layers = []
        self.loss = None
        self.loss_prime = None

    def __forward(self, input: list, training):
        output = input
        for layer in self.layers:
            output = layer.forward(output, training)
        return output

    def add_layer(self, layer: layer.Layer):
        self.layers.append(layer)

    def set_loss_functions(self, loss: callable, loss_prime: callable) -> None:
        self.__loss = loss
        self.__loss_prime = loss_prime

    def __train_batch(self, x_batch, y_batch, learning_rate):
        errors = []
        batch_loss = []
        for i, _ in enumerate(x_batch):
            output = self.__forward(x_batch[i], True)
            batch_loss.append(self.__loss(y_batch[i], output))
            errors.append(self.__loss_prime(y_batch[i], output))
        error = np.mean(errors, axis=0)
        for layer in reversed(self.layers):
            error = layer.backward(error)
        return np.mean(batch_loss)

    def __shuffle(self, x_train, y_train):
        zipped = list(zip(x_train, y_train))
        random.shuffle(zipped)
        x_train, y_train = zip(*zipped)
        return np.array(x_train), np.array(y_train)

    def train(self, x_train, y_train, epochs: int, batch_size: int = 16):
        if self.__loss is None or self.__loss_prime is None:
            raise NotImplementedError

        for epoch in tqdm.tqdm(range(epochs)):
            loss = []
            x_train_epoch, y_train_epoch = self.__shuffle(x_train, y_train)
            for i in range(0, len(x_train), 1): 
                #x_batch = x_train_epoch[i:i + batch_size]
                #y_batch = y_train_epoch[i:i + batch_size]
                #loss.append(self.__train_batch(x_batch, y_batch, 0.01))

                output = self.__forward(x_train_epoch[i], True)
                loss.append(self.__loss(y_train_epoch[i], output))

                error = self.__loss_prime(y_train_epoch[i], output)
                for layer in reversed(self.layers):
                    error = layer.backward(error)
            loss = np.mean(loss)
            tqdm.tqdm.write(f'Epoch: {epoch}, Loss: {loss}')

    def predict(self, input: set):
        return self.__forward(input, False)

    def save(self, filename: str):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def restore(filename: str):
        with open(filename, 'rb') as f:
            return pickle.load(f)
