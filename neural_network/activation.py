import numpy as np

def function_linear(input: float) -> float:
    return input

def function_linear_prime(input: float) -> float:
    return 1

linear = {
    "function": function_linear,
    "prime": function_linear_prime
}

def function_relu(inputs: float) -> float:
    for i, input in enumerate(inputs):
        if input < 0:
            inputs[i] = 0
    return inputs

def function_relu_prime(inputs: float) -> float:
    for i, input in enumerate(inputs):
        inputs[i] = 1 if input > 0 else 0
    return inputs

relu = {
    "function": function_relu,
    "prime": function_relu_prime
}

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
    "tanh": tanh,
    "relu": relu,
    "linear": linear
}
