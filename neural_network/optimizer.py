import numpy as np

class SGDOptimizer:
    def __init__(self, lr: float) -> None:
        self.lr = lr

    def update(self, weights: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        return weights - self.lr * gradients
