import numpy as np
import sys

def function_linear(input) -> float:
    return input

def function_linear_prime(input) -> float:
    return 1

linear = {
    "function": function_linear,
    "prime": function_linear_prime
}

def function_relu(inputs) -> float:
    return np.maximum(0, inputs)

def function_relu_prime(inputs) -> float:
    return np.where(inputs <= 0, 0, 1)

relu = {
    "function": function_relu,
    "prime": function_relu_prime
}

def function_sigmoid(input) -> float:
    return 1 / (1 + np.exp(-input))

def function_sigmoid_prime(input) -> float:
    return function_sigmoid(input) * (1 - function_sigmoid(input))

sigmoid = {
    "function": function_sigmoid,
    "prime": function_sigmoid_prime
}

def function_tanh(input) -> float:
    return np.tanh(input)

def tanh_prime(input) -> float:
    return 1 - np.tanh(input) ** 2

tanh = {
    "function": function_tanh,
    "prime": tanh_prime
}

activation_functions = {
    "sigmoid": sigmoid,
    "tanh": tanh,
    "relu": relu,
    "linear": linear,
    #"softmax": softmax
}
