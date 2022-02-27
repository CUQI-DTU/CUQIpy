import numpy as np
import inspect
from scipy.sparse import issparse
from cuqi.geometry import _DefaultGeometry
from dataclasses import dataclass

def force_ndarray(value,flatten=False):
    if not isinstance(value, np.ndarray) and value is not None and not issparse(value) and not callable(value):
        if hasattr(value,'__len__') and len(value)>1:
            value = np.array(value)
        else:
            value = np.array(value).reshape((1,1))
            
        if flatten is True:
            value = value.flatten()
    if isinstance(value,np.matrix): #Convert to array if matrix (matrix acts different on (n,) arrays)
        value = value.A
    return value


def getNonDefaultArgs(func):
    """ Returns the non-default arguments and kwargs from a callable function"""
    sig = inspect.signature(func)
    para = sig.parameters

    nonDefaultArgs = []
    for key in para:
        if key != "kwargs" and key != "args" and para[key].default is inspect._empty: #no default and not kwargs
            nonDefaultArgs.append(key)
    return nonDefaultArgs


def get_direct_attributes(dist):
    keys = vars(dist).keys()
    return [key for key in keys]


def get_indirect_attributes(dist):
    attributes = []
    for key, value in vars(dist).items():
        if callable(value):
            attributes.extend(getNonDefaultArgs(value))
    return attributes

def first_order_finite_difference_gradient(func, x, dim, epsilon= 0.000001):
    FD_gradient = np.empty(dim)
 
    for i in range(dim):
        eps_vec = np.zeros(dim)
        eps_vec[i] = epsilon
        x_plus_eps = x + eps_vec
        FD_gradient[i] = (func(x_plus_eps) - func(x))/epsilon
        
    return FD_gradient

@dataclass
class ProblemInfo:
    """Problem info dataclass. Gives a convenient way to store data defined in test-problems."""
    exactSolution: np.ndarray = None
    exactData: np.ndarray = None
    infoString: str = None

    def __repr__(self) -> str:
        out_str = "ProblemInfo with the following set attributes:\n"+str(self.getSetAttributes())      
        if self.infoString is not None:
            return out_str+"\n infoString: "+str(self.infoString)
        else:
            return out_str

    def getSetAttributes(self):
        """Returns a list of all attributes that are not None."""
        dict = vars(self)
        return list({key for key in dict if dict[key] is not None})