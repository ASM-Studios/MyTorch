import numpy as np

def function_cross_entropy(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred));

def function_cross_entropy_prime(y_true, y_pred):
    return -y_true / y_pred;

cross_entropy = {
    "function": function_cross_entropy,
    "prime": function_cross_entropy_prime
}

def function_mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2));

def function_mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / len(y_true);

mse = {
    "function": function_mse,
    "prime": function_mse_prime
}

loss_functions = {
    "mse": mse,
    "cross_entropy": cross_entropy
}
