#!/usr/bin/env python3

import sys
from neural_network import neural_network

class Config:
    def __init__(self):
        self.__nn_file = None
        self.__nn_savefile = None
        self.__cb_file = None
        self.__mode = 0

    def set_mode(self, mode):
        if (self.__mode != 0):
            print('Mode cannot be combined')
            sys.exit(84)
        if (mode == '--predict'):
            self.__mode = 1
        elif (mode == '--train'):
            self.__mode = 2
        else:
            print('Invalid mode')
            sys.exit(84)

    def set_nn_savefile(self, file):
        if (self.__mode == 1):
            print('Incompatible argument (save and predict)')
            sys.exit(84)
        self.__nn_savefile = file
        return

    def parse(self):
        try:
            self.set_mode(sys.argv[1])
            offset = 0
            if (sys.argv[2] == '--save'):
                self.set_nn_savefile(sys.argv[3])
                offset += 2
            self.__nn_file = sys.argv[2 + offset]
            self.__cb_file = sys.argv[3 + offset]
        except IndexError:
            print('Bad arguments')
            sys.exit(84)
        return

if __name__ == '__main__':
    config = Config()
    config.parse()

    sys.exit(0)
    nn = neural_network.NeuralNetwork.restore(sys.argv[1])
    print(len(nn.get_layers()))
    print(nn.predict([[[0,0]]]))
