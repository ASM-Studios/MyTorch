#!/usr/bin/env python3

import sys
import json
from neural_network import neural_network, activation, layer
from neural_network.activation import activation_functions
from neural_network.loss import loss_functions

def load_layer(data: json, nn: neural_network.NeuralNetwork) -> None:
    if (data['type'] == None):
        print('Layer type not specified')
        sys.exit(84)
    if (data['type'] == 'activation'):
        if (data.get('activation') == None):
            print('Invalid layer parameters')
            sys.exit(84)
        try:
            function = activation_functions[data['activation']]['function']
            function_prime = activation_functions[data['activation']]['prime']
            nn.add_layer(layer.ActivationLayer(function, function_prime))
        except KeyError:
            print(f'Activation function "{data['activation']}" does not exist')
            print(f'List of activation function: {list(activation_functions.keys())}')
            sys.exit(84)
    elif (data['type'] == 'fully_connected'):
        if (data.get('input_size') == None or data.get('output_size') == None):
            print('Invalid layer parameters')
            sys.exit(84)
        try:
            input_size = int(data['input_size'])
            output_size = int(data['output_size'])
            nn.add_layer(layer.FCLayer(input_size, output_size))
        except ValueError:
            print('Sizes must be of type int')
            sys.exit(84)
    else:
        print(f'Unkonwn layer type: {data['type']}')
        sys.exit(84)

def load_base(data: json) -> neural_network.NeuralNetwork:
    nn = neural_network.NeuralNetwork()
    if (data.get('loss') == None):
        print('Loss function not specified')
        sys.exit(84)
    try:
        function = loss_functions[data['loss']]['function']
        function_prime = loss_functions[data['loss']]['prime']
        nn.set_loss_functions(function, function_prime)
    except KeyError:
        print(f'Loss function "{data['loss']}" does not exist')
        print(f'List of loss function: {list(loss_functions.keys())}')
        sys.exit(84)
    return nn

def parse_nn_conf(data: json) -> neural_network.NeuralNetwork:
    nn = load_base(data)
    for layer in data['layers']:
        load_layer(layer, nn)
    return nn

def load_nn_conf(file: str, nb: int) -> None:
    try:
        with open(file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f'File {file} does not exist')
        sys.exit(84)
    except json.decoder.JSONDecodeError as exception:
        print(f'File {file} is invalid (line {exception.lineno})')
        sys.exit(84)
    nn = parse_nn_conf(data)
    for i in range(nb):
        file = file.split('.')[0]
        nn.save(f'{file}_{i + 1}.nn')


if __name__ == '__main__':
    for i in range(1, len(sys.argv), 2):
        try:
            file = sys.argv[i]
            nb = int(sys.argv[i + 1])
        except IndexError as e:
            print(f"Missing argument for {sys.argv[i]}")
            sys.exit(84)
        except ValueError as e:
            print(f"{sys.argv[i + 1]} is not a valid number")
            sys.exit(84)
        load_nn_conf(file, nb)
    sys.exit(0)
