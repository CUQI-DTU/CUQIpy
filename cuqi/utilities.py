import numpy as np
import inspect
from scipy.sparse import issparse
from cuqi.geometry import _DefaultGeometry
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

def check_geometries_consistency(geom1, geom2, fail_msg):
    """checks geom1 and geom2 consistency . If both are of type `_DefaultGeometry` they need to be equal. If one of them is of `_DefaultGeometry` type, it will take the value of the other one. If both of them are user defined, they need to be consistent"""
    if isinstance(geom1,_DefaultGeometry):
        if isinstance(geom2,_DefaultGeometry):
            if geom1 == geom2:
                return geom1,geom2
        else: 
            return geom2, geom2
    else:
        if isinstance(geom2,_DefaultGeometry):
            return geom1,geom1
        else:
            if geom1 == geom2:
                return geom1,geom2
    raise Exception(fail_msg)


def get_direct_attributes(dist):
    keys = vars(dist).keys()
    return [key for key in keys]


def get_indirect_attributes(dist):
    attributes = []
    for key, value in vars(dist).items():
        if callable(value):
            attributes.extend(getNonDefaultArgs(value))
    return attributes

