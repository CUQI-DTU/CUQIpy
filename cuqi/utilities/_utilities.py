from cuqi.samples import CUQIarray
import numpy as np
import inspect
from numbers import Number
from scipy.sparse import issparse, diags
from scipy.sparse import linalg as spslinalg
from dataclasses import dataclass
from abc import ABCMeta


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

def infer_len(value):
    """ Infer the length of the given input value.

    Matrices are assumed to have length equal to the number of rows.
    Numbers are considered to have length 1.
    Other objects with no length are considered to have length 0.
    """
    if hasattr(value,'__len__'):
        try:
            return len(value)
        except TypeError: #Special-case for scipy sparse matrices, which have len but return an error
            return value.shape[0]
    elif isinstance(value, Number):
        return 1
    else:
        return 0

def get_non_default_args(func):
    """ Returns the non-default arguments and kwargs from a callable function"""
    # If the function has variable _non_default_args, use that for speed.
    if hasattr(func, '_non_default_args'):
        return func._non_default_args

    # Otherwise, get the arguments from the function signature.
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

def get_indirect_variables(dist):
    attributes = []
    for _, value in vars(dist).items():
        if callable(value):
            keys = get_non_default_args(value)
            for key in keys:
                if key not in attributes: #Ensure we did not already find this key
                    attributes.append(key)
    return attributes

def get_writeable_attributes(dist):
    """ Get writeable attributes of object instance. """
    attributes = []
    for key in vars(dist).keys():
        if key[0] != "_":
            attributes.append(key)
    return attributes

def get_writeable_properties(cls, stop_at_class=object):
    """ Get writeable properties of class type."""

    # Potentially convert object instance to class type.
    if isinstance(cls, stop_at_class) and isinstance(type(cls), ABCMeta):
        cls = type(cls)

    # Compute writeable properties of this class
    writeable_properties = [attr for attr, value in vars(cls).items()
                 if isinstance(value, property) and value.fset is not None]

    # Stop recursion at stop_at_class
    if cls == stop_at_class:
        return writeable_properties

    # Recursively get writeable properties of parents
    for base in cls.__bases__:
        writeable_properties += get_writeable_properties(base)
    return writeable_properties

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
    Miscellaneous: dict = None
    infoString: str = None

    def __repr__(self) -> str:
        out_str = "ProblemInfo with the following set attributes:\n"+str(self.getSetAttributes())      
        if self.infoString is not None:
            out_str = out_str+"\n infoString: "+str(self.infoString)
        if self.Miscellaneous is not None:
            out_str = out_str+f"\n Miscellaneous: {self.Miscellaneous.keys()}"
        return out_str

    def getSetAttributes(self):
        """Returns a list of all attributes that are not None."""
        dict = vars(self)
        return list({key for key in dict if dict[key] is not None})

# work-around to compute sparse Cholesky
def sparse_cholesky(A):
    """Computes Cholesky factorization for sparse matrix `A` and returns the upper triangular factor `U`, where `A=U^T@U`"""
    # https://gist.github.com/omitakahiro/c49e5168d04438c5b20c921b928f1f5d
    LU = spslinalg.splu(A, diag_pivot_thresh=0, permc_spec='natural') # sparse LU decomposition


    # check the matrix A is positive definite
    if (LU.perm_r == np.arange(A.shape[0])).all() and (LU.U.diagonal() > 0).all(): 
        return (LU.L @ (diags(LU.U.diagonal()**0.5))).T
    else:
        raise TypeError('The matrix is not positive semi-definite')

def approx_derivative(func, wrt, direction=None, epsilon=np.sqrt(np.finfo(np.float).eps)):
    """Approximates the derivative of callable (possibly vector-valued) function `func` evaluated at point `wrt`. If `direction` is provided, the direction-Jacobian product will be computed and returned, otherwise, the Jacobian matrix (or the gradient in case of a scalar function `func`) will be returned. The approximation is done using forward differences.

    Parameters
    ----------
    func: function handler
        A vector-valued function of the form func(x).

    wrt : ndarray
        The point at which the derivative to be evaluated.

    direction : ndarray
        The direction used to compute direction-Jacobian product. 
        If None, the Jacobian matrix is returned.

    epsilon: float
        The spacing in the finite difference approximation.

    Returns
    -------
    ndarray
        The approximate Jacobian matrix.
    """
    # Raise an error if wrt or direction is a CUQIarray.
    # Example of scenario where this is needed: 
    # the line Matr[i] = (func(wrt+dx) - f0)/epsilon
    # does not give correct results if for example
    # wrt is a CUQIarray with is_par=False and its 
    # corresponding geometry par2fun map is not identity
    # (e.g. funvalues=paramters**2), because wrt entries 
    # are interpreted as function value.
    if isinstance(wrt, CUQIarray) or isinstance(direction, CUQIarray):
        raise NotImplementedError("approx_derivative is not implemented"+
                                  "for inputs of type CUQIarray")
    wrt = np.asfarray(wrt)
    f0 = func(wrt)
    Matr = np.zeros([len(wrt), len(f0)])
    dx = np.zeros(len(wrt))
    for i in range(len(wrt)):
        dx[i] = epsilon
        Matr[i] = (func(wrt+dx) - f0)/epsilon
        dx[i] = 0.0
    if direction is None:
        return Matr.T
    else:
        return Matr@direction