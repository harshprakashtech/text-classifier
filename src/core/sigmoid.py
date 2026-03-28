import numpy as np

# Helper function to calculate the sigmoid
# It squishes any value between 0 and 1
# Sigmoid function: f(x) = 1 / (1 + e^(-x))

def get_sigmoid(value: float) -> float:
    return 1 / (1 + np.exp(-value))