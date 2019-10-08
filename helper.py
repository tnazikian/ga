import numpy as np

def sigmoid_gyaku(x):
    """Takes in a fitness score and returns a value between 1 and 0"""
    return 2/(1+np.exp(x))