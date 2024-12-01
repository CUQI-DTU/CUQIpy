import numpy as np
from cuqi.geometry import _DefaultGeometry1D


class CUQIarray(np.ndarray):
    """
    A class to represent data arrays, subclassed from numpy array, along with geometry and plotting

    Parameters
    ----------
    input_array : ndarray
        A numpy array holding the parameter or function values. 
    
    is_par : bool, default True
        Boolean flag whether input_array is to be interpreted as parameter (True) or function values (False).

    geometry : cuqi.geometry.Geometry, default None
        Contains the geometry related of the data.
    """

    def __repr__(self) -> str: 
        return "CUQIarray: NumPy array wrapped with geometry.\n" + \
               "---------------------------------------------\n\n" + \
            "Geometry:\n {}\n\n".format(self.geometry) + \
            "Parameters:\n {}\n\n".format(self.is_par) + \
            "Array:\n" + \
            super().__repr__()

    def __new__(cls, input_array, is_par=True, geometry=None):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # add the new attribute to the created instance
        obj.is_par = is_par
        if (not is_par) and (geometry is None):
            raise ValueError("geometry cannot be none when initializing a CUQIarray as function values (with is_par False).")
        if is_par and (obj.ndim>1):
            raise ValueError("input_array cannot be multidimensional when initializing CUQIarray as parameter (with is_par True).")
        if geometry is None:
            geometry = _DefaultGeometry1D(grid=obj.__len__())
        obj.geometry = geometry
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None: return
        self.is_par = getattr(obj, 'is_par', True)
        self.geometry = getattr(obj, 'geometry', None)

    @property
    def funvals(self):
        """ Returns itself as function values. """
        if self.is_par is True:
            vals = self.geometry.par2fun(self)
        else:
            vals = self

        if isinstance(vals, np.ndarray):
            if vals.dtype == np.dtype('O'):
                # if vals is of type np.ndarray, but the data type of the array
                # is object (e.g. FEniCS function), then extract the object and
                # return it. reshape(1) is needed to convert the shape from
                # () to (1,).
                return self.reshape(1)[0]
            else:
                # else, cast the np.ndarray to a CUQIarray
                return type(self)(vals,is_par=False,geometry=self.geometry) #vals.view(np.ndarray)
        else:
            return vals 

    @property
    def parameters(self):
        """ Returns itself as parameters. """
        if self.is_par is False:
            if self.dtype == np.dtype('O'):
                # If the current state if the CUQIarray is function values, and
                # the data type of self is object (e.g. FEniCS function), then
                # extract the object and save it. reshape(1) is needed to
                # convert the shape from () to (1,).
                funvals = self.reshape(1)[0]
            else:
                funvals = self
            vals = self.geometry.fun2par(funvals)

        else:
            vals = self
        return type(self)(vals,is_par=True,geometry=self.geometry)

    def to_numpy(self):
        """Return a numpy array of the CUQIarray data. If is_par is True, then 
        the parameters are returned as numpy.ndarray. If is_par is False, then 
        the function values are returned instead.
        """
        try:
            return self.view(np.ndarray)
        except:
            raise ValueError(
                f"Cannot convert {self.__class__.__name__} to numpy array")

    def plot(self, plot_par=False, **kwargs):
        """ Plot the data as function or parameters. """
        if plot_par:
            kwargs["is_par"]=True
            return self.geometry.plot(self.parameters, plot_par=plot_par, **kwargs)
        else:
            kwargs["is_par"]=False
            return self.geometry.plot(self.funvals, **kwargs)
