import numpy as np

def force_ndarray(value):
    if not isinstance(value, np.ndarray):
        value = np.array([value]).reshape((1,1))
    return value