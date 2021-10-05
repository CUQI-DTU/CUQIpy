import numpy as np
from scipy.sparse import issparse
def force_ndarray(value,flatten=False):
    if not isinstance(value, np.ndarray) and value is not None and not issparse(value) and not callable(value):
        value = np.array(value).reshape((1,1))
        if flatten is True:
            value = value.flatten()
    return value