import numpy as np

def force_ndarray(value):
    if not isinstance(value, np.ndarray) and value is not None:
        value = np.array([value]).reshape((1,1))
    return value