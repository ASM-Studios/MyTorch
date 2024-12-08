import numpy as np

def function_cross_entropy(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred))

def function_softmax_cross_entropy_prime(y_true, y_pred):
    return y_pred - y_true

cross_entropy = {
    'function': function_cross_entropy,
    'prime': function_softmax_cross_entropy_prime
}

def function_mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2));

def function_mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / len(y_true);

mse = {
    'function': function_mse,
    'prime': function_mse_prime
}

loss_functions = {
    'mse': mse,
    'cross_entropy': cross_entropy
}
