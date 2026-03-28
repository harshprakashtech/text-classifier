import numpy as np

# Loss function to calculate the error
def get_loss(prediction, truth):
    return -(truth * np.log(prediction) + (1 - truth) * np.log(1 - prediction))