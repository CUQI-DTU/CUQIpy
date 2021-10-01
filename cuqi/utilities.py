import numpy as np
from scipy.sparse import issparse
def force_ndarray(value):
    if not isinstance(value, np.ndarray) and value is not None and not issparse(value):
        value = np.array(value).reshape((1,1))
    return value