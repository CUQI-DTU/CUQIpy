from cuqi.array import CUQIarray
from cuqi.density import Density
import numpy as np
import inspect
from numbers import Number
from scipy.sparse import issparse, diags
from scipy.sparse import linalg as spslinalg
from dataclasses import dataclass
from abc import ABCMeta
import copy
import matplotlib.pyplot as plt


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
    for attribute in dist.get_mutable_variables():
        value = getattr(dist, attribute)
        if check_if_conditional_from_attr(value):
            keys = get_non_default_args(value)
            for key in keys:
                if key not in attributes: #Ensure we did not already find this key
                    attributes.append(key)
    return attributes 

def check_if_conditional_from_attr(value):
    """
    Check if a distribution is conditional from a given attribute.
    So far, we assume that a distribution is conditional if
    - the given attribute is a callable function and
    - the given attribute is not a LinearOperator.
    """
    if isinstance(value, spslinalg.LinearOperator):
        return False
    elif callable(value):
        return True
    else:
        return False

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

def approx_derivative(func, wrt, direction=None, epsilon=np.sqrt(np.finfo(float).eps)):
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

    # We compute the Jacobian matrix of func using forward differences.
    # If the function is scalar-valued, we compute the gradient instead.
    # If the direction is provided, we compute the direction-Jacobian product.
    wrt = np.asfarray(wrt)
    f0 = func(wrt)
    Matr = np.zeros([infer_len(wrt), infer_len(f0)])
    dx = np.zeros(len(wrt))

    # Compute the Jacobian matrix (transpose)
    for i in range(len(wrt)):
        dx[i] = epsilon
        Matr[i] = (func(wrt+dx) - f0)/epsilon
        dx[i] = 0.0

    # Return the Jacobian matrix (or the gradient)
    # or the direction-Jacobian product
    if direction is None:
        if infer_len(f0) == 1:
            return Matr.reshape(infer_len(wrt))
        else:
            return Matr.T
    else:
        return Matr@direction

def approx_gradient(func, x, epsilon= 0.000001):
    """Approximates the gradient of callable scalar function `func` evaluated at point `x`. The approximation is done using finite differences with
    step size `epsilon`."""
    
    # Derivative of a scalar function using forward differences
    if isinstance(x, Number):
        return (func(x+epsilon) - func(x))/epsilon

    # Initialize variables
    FD_gradient = x*0.0
    eps_vec = x*0.0
    func_x = func(x)

    # Compute the gradient using forward differences component by component
    x_len = infer_len(x)
    for i in range(x_len):
        eps_vec[i] = epsilon
        x_plus_eps = x + eps_vec
        FD_gradient[i] = (func(x_plus_eps) - func_x)/epsilon
        eps_vec[i] = 0.0
        
    return FD_gradient


# Function for plotting 1D density functions
def plot_1D_density(density:Density,
                    v_min, v_max,
                    N=501, log_scale=False,
                    **kwargs):
    """ Plot 1D density function 

    Parameters
    ----------
    density : CUQIpy Density
        The density to be plotted.

    v_min : float
        Minimum value for the variable.
    
    v_max : float
        Maximum value for the variable.

    N : int
        Number of grid points for the variable.

    log_scale : bool
        If True, the density is plotted in log scale.
    
    kwargs : dict
        Additional keyword arguments for the plot that are passed to the
        underlying plotting method: `matplotlib.pyplot.plot` function 
        in this case.

    """
    # Assert that the density is 1D
    assert density.dim == 1, "The density must be for a scalar variable"
    ls = np.linspace(v_min, v_max, N)

    # Create a map to evaluate density
    density_map = (lambda x: x) if log_scale else (lambda x: np.exp(x))

    # Evaluate density on grid
    y = [density_map(density.logd(grid_point)) for grid_point in ls]
    p = plt.plot(ls, y, **kwargs)
    return p

# Function for plotting 2D density functions
def plot_2D_density(density: Density, 
                    v1_min, v1_max,
                    v2_min, v2_max,
                    N1=201, N2=201,
                    log_scale=False,
                    **kwargs):
    """ Plot 2D density function 

    Parameters
    ----------
    density : CUQIpy Density
        The density to be plotted.

    v1_min : float
        Minimum value for the first variable.
    
    v1_max : float
        Maximum value for the first variable.
    
    v2_min : float
        Minimum value for the second variable.

    v2_max : float  
        Maximum value for the second variable.

    N1 : int
        Number of grid points for the first variable.

    N2 : int
        Number of grid points for the second variable.
    
    log_scale : bool
        If True, the density is plotted in log scale.

    kwargs : dict
        Additional keyword arguments for the plot that are passed to the
        underlying plotting method: `matplotlib.pyplot.imshow` function 
        in this case.

    """
    # Assert that the density is 2D
    assert density.dim == 2,\
        "The density must be for a two-dimensional variable"
    # Create grid
    ls1 = np.linspace(v1_min, v1_max, N1)
    ls2 = np.linspace(v2_min, v2_max, N2)
    grid1, grid2 = np.meshgrid(ls1, ls2)

    # Create a map to evaluate density
    density_map = (lambda x: x) if log_scale else (lambda x: np.exp(x))

    # Evaluate density on grid
    evaluated_density = np.zeros((N1, N2))
    for ii in range(N1):
        for jj in range(N2):
            evaluated_density[ii,jj] = density_map(
                density.logd([grid1[ii,jj], grid2[ii,jj]])) 

    # Plot
    pixelwidth_x = (v1_max-v1_min)/(N2-1)
    pixelwidth_y = (v2_max-v2_min)/(N2-1)

    hp_x = 0.5*pixelwidth_x
    hp_y = 0.5*pixelwidth_y

    extent = (v1_min-hp_x, v1_max+hp_x, v2_min-hp_y, v2_max+hp_y)

    im = plt.imshow(evaluated_density, origin='lower', extent=extent, **kwargs)
    return im



def count_nonzero(x, threshold = 1e-6):
        """ Returns the number of values in an array whose absolute value is larger than a specified threshold

        Parameters
        ----------
        x : `np.ndarray` 
            Array to count nonzero elements of.

        threshold : float
            Theshold for considering a value as nonzero.
        """
        return np.sum([np.abs(v) >= threshold for v in x])
    
def count_constant_components_1D(x, threshold = 1e-2, lower = -np.inf, upper = np.inf):
        """ Returns the number of piecewise constant components in a one-dimensional array

        Parameters
        ----------
        x : `np.ndarray` 
            1D Array to count components of.

        threshold : float
            Strict theshold on when the difference of neighbouring values is considered zero.

        lower : float
            Piecewise constant components below this value are not counted.

        upper : float
            Piecewise constant components above this value are not counted.
        """

        counter = 0
        if x[0] > lower and x[0] < upper:
            counter += 1
        
        x_previous = x[0]

        for x_current in x[1:]:
            if (abs(x_previous - x_current) >= threshold and
                x_current > lower and
                x_current < upper):
                    counter += 1

            x_previous = x_current
    
        return counter
        
def count_constant_components_2D(x, threshold = 1e-2, lower = -np.inf, upper = np.inf):
        """ Returns the number of piecewise constant components in a two-dimensional array

        Parameters
        ----------
        x : `np.ndarray` 
            2D Array to count components of.

        threshold : float
            Strict theshold on when the difference of neighbouring values is considered zero.

        lower : float
            Piecewise constant components below this value are not counted.

        upper : float
            Piecewise constant components above this value are not counted.
        """
        filled = np.zeros_like(x, dtype = int)
        counter = 0

        def process(i, j):
            queue = []
            queue.append((i,j))
            filled[i, j] = 1
            while len(queue) != 0:
                (icur, jcur) = queue.pop(0)
                
                if icur > 0 and filled[icur - 1, jcur] == 0 and abs(x[icur, jcur] - x[icur - 1, jcur]) <= threshold:
                    filled[icur - 1, jcur] = 1
                    queue.append((icur-1, jcur))
                if jcur > 0 and filled[icur, jcur-1] == 0 and abs(x[icur, jcur] - x[icur, jcur - 1]) <= threshold:
                    filled[icur, jcur-1] = 1
                    queue.append((icur, jcur-1))
                if icur < x.shape[0]-1 and filled[icur + 1, jcur] == 0 and abs(x[icur, jcur] - x[icur + 1, jcur]) <= threshold:
                    filled[icur + 1, jcur] = 1
                    queue.append((icur+1, jcur))
                if jcur < x.shape[1]-1 and filled[icur, jcur + 1] == 0 and abs(x[icur, jcur] - x[icur, jcur + 1]) <= threshold:
                    filled[icur, jcur + 1] = 1
                    queue.append((icur, jcur+1))
        
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                if filled[i,j] == 0:
                    if x[i,j] > lower and x[i,j] < upper:
                        counter += 1
                    process(i, j)
        return counter
                    