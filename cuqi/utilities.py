import numpy as np
import inspect
from scipy.sparse import issparse
def force_ndarray(value,flatten=False):
    if not isinstance(value, np.ndarray) and value is not None and not issparse(value) and not callable(value):
        if hasattr(value,'__len__') and len(value)>1:
            value = np.array(value)
        else:
            value = np.array(value).reshape((1,1))
            
        if flatten is True:
            value = value.flatten()
    return value


def getNonDefaultArgs(func):
    """ Returns the non-default arguments and kwargs from a callable function"""
    sig = inspect.signature(func)
    para = sig.parameters

    nonDefaultArgs = []
    for key in para:
        if para[key].default is inspect._empty: #no default
            nonDefaultArgs.append(key)

    return nonDefaultArgs