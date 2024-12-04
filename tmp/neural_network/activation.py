import numpy as np

def relu(input: float) -> float:
    return input if input > 0 else 0

def sigmoid(input: float) -> float:
    return 1 / (1 + np.exp(-input))

def sigmoid_derivative(input: float) -> float:
    return sigmoid(input) * (1 - sigmoid(input))

def heaviside_function(input: float) -> int:
    return 1 if input > 0 else 0

def tanh(input: float) -> float:
    return np.tanh(input)

def tanh_prime(input: float) -> float:
    return 1 - np.tanh(input) ** 2
