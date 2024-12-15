#!/usr/bin/env python3

import sys
from neural_network import neural_network
import numpy as np

class Config:
    def __init__(self):
        self.nn_file = None
        self.nn_savefile = None
        self.cb_file = None
        self.mode = 0

    def set_mode(self, mode):
        if self.mode != 0:
            print('Mode cannot be combined')
            sys.exit(84)
        if mode == '--predict':
            self.mode = 1
        elif mode == '--train':
            self.mode = 2
        else:
            print('Invalid mode')
            sys.exit(84)

    def set_nn_savefile(self, file):
        if self.mode == 1:
            print('Incompatible argument (save and predict)')
            sys.exit(84)
        self.nn_savefile = file

    def parse(self):
        try:
            self.set_mode(sys.argv[1])
            offset = 0
            if sys.argv[2] == '--save':
                self.set_nn_savefile(sys.argv[3])
                offset += 2
            self.nn_file = sys.argv[2 + offset]
            self.cb_file = sys.argv[3 + offset]
        except IndexError:
            print('Bad arguments')
            sys.exit(84)

pieces = {
    'P': 0,
    'N': 1,
    'B': 2,
    'R': 3,
    'Q': 4,
    'K': 5,
    'p': 6,
    'n': 7,
    'b': 8,
    'r': 9,
    'q': 10,
    'k': 11
}

result = {
    'Nothing': 0,
    'Check White': 1,
    'Check Black': 2,
    'Checkmate White': 3,
    'Checkmate Black': 4,
    'Stalemate': 5
}

def get_plate(plate: str) -> np.ndarray:
    matrix = np.zeros((12, 8, 8), dtype=int)
    x = 0
    y = 0
    for character in plate:
        if character == '/':
            y += 1
            x = 0
            continue
        if character.isdigit():
            x += int(character)
            continue
        matrix[pieces[character]][y][x] = 1
        x += 1
    return matrix

def get_castling(castling_config: str) -> np.ndarray:
    castling = np.zeros(4, dtype = int)
    if castling_config == '-':
        return castling
    castling[0] = 1 if 'K' in castling_config else 0
    castling[1] = 1 if 'Q' in castling_config else 0
    castling[2] = 1 if 'k' in castling_config else 0
    castling[3] = 1 if 'q' in castling_config else 0
    return castling

def get_en_passant(en_passant_config: str) -> np.ndarray:
    en_passant = np.zeros((8, 8), dtype = int)
    if en_passant_config == '-':
        return en_passant
    x = ord(en_passant_config[0]) - ord('a')
    y = int(en_passant_config[1]) - 1
    en_passant[y][x] = 1
    return en_passant.flatten()

def get_input(chess_config: str) -> np.ndarray:
    chess_config = chess_config.split(' ')
    matrix = get_plate(chess_config[0]).flatten()
    matrix = np.append(matrix, 1 if chess_config[1] == 'w' else 0)
    matrix = np.append(matrix, get_castling(chess_config[2]))
    #matrix = np.append(matrix, get_en_passant(chess_config[3]))
    return np.array([matrix], dtype=float)

def get_output(chess_config: str) -> np.ndarray:
    y = np.zeros(6, dtype=float)
    array = chess_config.split(' ')
    if array[6] == 'Check' or array[6] == 'Checkmate':
        y[result[array[6] + ' ' + array[7]]] = 1
    else:
        y[result[chess_config.split(' ')[6]]] = 1
    return np.array([y], dtype=float)

def train(config: Config, nn: neural_network.NeuralNetwork):
    with open(config.cb_file, 'r') as f:
        data = f.read()
    data = data.split('\n')
    x_train = []
    y_train = []
    for i, chess_config in enumerate(data):
        if chess_config == '':
            continue
        x_train.append(get_input(chess_config))
        y_train.append(get_output(chess_config))
    nn.train(x_train, y_train, 2, 16)

def predict(config: Config, nn:neural_network.NeuralNetwork):
    with open(config.cb_file, 'r') as f:
        data = f.read()
    data = data.split('\n')
    for chess_config in data:
        if chess_config == '':
            print('')
            continue
        index = np.argmax(nn.predict(get_input(chess_config)))
        print(list(result.keys())[index])

def execute(config: Config, nn: neural_network.NeuralNetwork):
    if config.mode == 1:
        predict(config, nn)
    elif config.mode == 2:
        train(config, nn)

def save(config: Config, nn: neural_network.NeuralNetwork):
    if config.nn_savefile is None:
        nn.save(config.nn_file)
    else:
        nn.save(config.nn_savefile)

if __name__ == '__main__':
    config = Config()
    config.parse()
    try:
        nn = neural_network.NeuralNetwork.restore(config.nn_file)
    except FileNotFoundError:
        print(f'Invalid file {config.nn_file}')
        sys.exit(84)
    execute(config, nn)
    if config.mode == 2:
        save(config, nn)
    sys.exit(0)
