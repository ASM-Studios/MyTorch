import numpy as np

def function_mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2));

def function_mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / 1;

mse = {
    "function": function_mse,
    "prime": function_mse_prime
}

loss_functions = {
    "mse": mse
}
