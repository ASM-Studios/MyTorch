import numpy as np

def relu(input: float) -> float:
    return input if input > 0 else 0

def heaviside_function(input: float) -> int:
    return 1 if input > 0 else 0

def function_sigmoid(input: float) -> float:
    return 1 / (1 + np.exp(-input))

def function_sigmoid_prime(input: float) -> float:
    return function_sigmoid(input) * (1 - function_sigmoid(input))

sigmoid = {
    "function": function_sigmoid,
    "prime": function_sigmoid_prime
}

def function_tanh(input: float) -> float:
    return np.tanh(input)

def tanh_prime(input: float) -> float:
    return 1 - np.tanh(input) ** 2

tanh = {
    "function": function_tanh,
    "prime": tanh_prime
}

activation_functions = {
    "sigmoid": sigmoid,
    "tanh": tanh
}
