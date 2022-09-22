import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse import hstack
from scipy.linalg import solve
from cuqi.samples import Samples, CUQIarray
from cuqi.geometry import Geometry, _DefaultGeometry, _get_identity_geometries
import cuqi
import matplotlib.pyplot as plt
from copy import copy

class Model(object):
    """Generic model defined by a forward operator.

    Parameters
    -----------
    forward : 2D ndarray or callable function.
        Forward operator.

    range_geometry : integer or cuqi.geometry.Geometry
        If integer is given a _DefaultGeometry is created with dimension of the integer.

    domain_geometry : integer or cuqi.geometry.Geometry
        If integer is given a _DefaultGeometry is created with dimension of the integer.

    gradient : callable function
        The direction-Jacobian product of the forward operator Jacobian with 
        respect to the forward operator input, evaluated at a point (`wrt`).
        The signature of the gradient function should be (`direction`, `wrt`),
        where `direction` is the direction by which the Jacobian matrix is
        multiplied and `wrt` is the point at which the Jacobian is computed.

    Attributes
    -----------
    range_geometry : cuqi.geometry.Geometry
        The geometry representing the range.

    domain_geometry : cuqi.geometry.Geometry
        The geometry representing the domain.

    Methods
    -----------
    :meth:`forward` the forward operator.
    :meth:`range_dim` the dimension of the range.
    :meth:`domain_dim` the dimension of the domain.
    """
    def __init__(self, forward, range_geometry, domain_geometry, gradient=None):

        #Check if input is callable
        if callable(forward) is not True:
            raise TypeError("Forward needs to be callable function of some kind")
        
        #Check if input is callable
        if (gradient is not None) and (callable(gradient) is not True):
            raise TypeError("Gradient needs to be callable function of some kind")
 
        #Store forward func
        self._forward_func = forward
        self._gradient_func = gradient
         
        #Store range_geometry
        if isinstance(range_geometry, int):
            self.range_geometry = _DefaultGeometry(grid=range_geometry)
        elif isinstance(range_geometry, Geometry):
            self.range_geometry = range_geometry
        elif range_geometry is None:
            raise AttributeError("The parameter 'range_geometry' is not specified by the user and it connot be inferred from the attribute 'forward'.")
        else:
            raise TypeError("The parameter 'range_geometry' should be of type 'int' or 'cuqi.geometry.Geometry'.")

        #Store domain_geometry
        if isinstance(domain_geometry, int):
            self.domain_geometry = _DefaultGeometry(grid=domain_geometry)
        elif isinstance(domain_geometry, Geometry):
            self.domain_geometry = domain_geometry
        elif domain_geometry is None:
            raise AttributeError("The parameter 'domain_geometry' is not specified by the user and it connot be inferred from the attribute 'forward'.")
        else:
            raise TypeError("The parameter 'domain_geometry' should be of type 'int' or 'cuqi.geometry.Geometry'.")

        # Store non_default_args of the forward operator for faster caching when checking for those arguments.
        self._non_default_args = cuqi.utilities.get_non_default_args(self)

    @property
    def domain_dim(self): 
        return self.domain_geometry.par_dim

    @property
    def range_dim(self): 
        return self.range_geometry.par_dim

    def _input2fun(self, x, geometry, is_par):
        """ Converts input to function values (if needed) using the appropriate geometry. The input can then be passed to :class:`~cuqi.model.Model` operators (e.g. _forward_func, _adjoint_func, _gradient_func).

        Parameters
        ----------
        x : ndarray or cuqi.samples.CUQIarray
            The input value to be converted.

        geometry : cuqi.geometry.Geometry
            The geometry representing the input `x`.

        is_par : bool
            If True the input is assumed to be parameters.
            If False the input is assumed to be function values.

        Returns
        -------
        ndarray or cuqi.samples.CUQIarray
            The input value represented as a function.
        """
        if type(x) is CUQIarray and not isinstance(x.geometry, _DefaultGeometry):
            return x.funvals
        elif is_par:
            return geometry.par2fun(x)
        else:
            return x

    def _output2par(self, out, geometry, to_CUQIarray=False):
        """ Converts output of :class:~`cuqi.model.Model` operators (e.g. _forward_func, _adjoint_func, _gradient_func) to parameters using the appropriate geometry.

        Parameters
        ----------
        out : ndarray or cuqi.samples.CUQIarray
            The output value to be converted.

        geometry : cuqi.geometry.Geometry
            The geometry representing the argument `out`.

        to_CUQIarray : bool
            If True, the output is wrapped as a cuqi.samples.CUQIarray.

        Returns
        -------
        ndarray or cuqi.samples.CUQIarray
            The output value represented as parameters.
        """ 
        out = geometry.fun2par(out)
        if to_CUQIarray:
            return CUQIarray(out, is_par=True, geometry=geometry)
        else:
            return out

    def _apply_func(self, func, func_range_geometry, func_domain_geometry, x, is_par, **kwargs):
        """ Private function that applies the given function `func` to the input value `x`. It converts the input to function values (if needed) using the given `func_domain_geometry` and converts the output function values to parameters using the given `func_range_geometry`. It additionally handles the case of applying the function `func` to the cuqi.samples.Samples object.

        kwargs are keyword arguments passed to the functions `func`.
        
        Parameters
        ----------
        func: function handler 
            The function to be applied.

        func_range_geometry : cuqi.geometry.Geometry
            The geometry representing the function `func` range.

        func_domain_geometry : cuqi.geometry.Geometry
            The geometry representing the function `func` domain.

        x : ndarray or cuqi.samples.CUQIarray
            The input value to the operator.

        is_par : bool
            If True the input is assumed to be parameters.
            If False the input is assumed to be function values.

        Returns
        -------
        ndarray or cuqi.samples.CUQIarray
            The output of the function `func` converted to parameters.
        """ 
        # If input x is Samples we apply func for each sample
        # TODO: Check if this can be done all-at-once for computational speed-up
        if isinstance(x,Samples):
            out = np.zeros((func_range_geometry.par_dim, x.Ns))
            # Recursively apply func to each sample
            for idx, item in enumerate(x):
                out[:,idx] = self._apply_func(func,
                                              func_range_geometry,
                                              func_domain_geometry,
                                              item, is_par=True,
                                              **kwargs)
            return Samples(out, geometry=func_range_geometry)

        x = self._input2fun(x, func_domain_geometry, is_par)
        out = func(x, **kwargs)
        return self._output2par(out, func_range_geometry, 
                                    to_CUQIarray= (type(x) is CUQIarray)) 
        
    def forward(self, x, is_par=True ):
        """ Forward function of the model.
        
        Forward converts the input to function values (if needed) using the domain geometry of the model.
        Forward converts the output function values to parameters using the range geometry of the model.

        Parameters
        ----------
        x : ndarray or cuqi.samples.CUQIarray
            The model input.

        is_par : bool
            If True the input is assumed to be parameters.
            If False the input is assumed to be function values.

        Returns
        -------
        ndarray or cuqi.samples.CUQIarray
            The model output. Always returned as parameters.
        """
        # If input is a distribution, we simply change the parameter name of model to match the distribution name
        if isinstance(x, cuqi.distribution.Distribution):
            if x.dim != self.domain_dim:
                raise ValueError("Attempting to match parameter name of Model with given distribution, but distribution dimension does not match model domain dimension.")
            new_model = copy(self)
            new_model._non_default_args = [x.name] # Defaults to x if distribution had no name
            return new_model

        # Else we apply the forward operator
        return self._apply_func(self._forward_func,
                                self.range_geometry,
                                self.domain_geometry,
                                x, is_par)

    def __call__(self,x):
        return self.forward(x)

    def gradient(self, direction, wrt, is_direction_par=True, is_wrt_par=True):
        """ Gradient of the forward operator.

        For non-linear models the gradient is computed using the
        forward operator and the Jacobian of the forward operator.

        Parameters
        ----------
        direction : ndarray
            The direction to compute the gradient. The Jacobian is applied to this direction.

        wrt : ndarray
            The point to compute the Jacobian at. This is only used for non-linear models.

        is_direction_par : bool
            If True, `direction` is assumed to be parameters.
            If False, `direction` is assumed to be function values.

        is_wrt_par : bool
            If True, `wrt` is assumed to be parameters.
            If False, `wrt` is assumed to be function values.
        
        """
        # Raise an error if _gradient_func function is not set
        if self._gradient_func is None:
            raise NotImplementedError("Gradient is not implemented for this model.")
        
        # Raise error if either the direction or wrt are Samples object
        if isinstance(direction, Samples) or isinstance(wrt, Samples):
            raise ValueError("cuqi.samples.Samples input values for arguments `direction` and `wrt` are not supported")
        
        # Raise an error if range_geometry is not in the list returned by
        # `_get_identity_geometries()`. i.e. The Jacobian of its 
        # par2fun map is not identity.  
        #TODO: Add range geometry gradient to the chain rule 
        if not type(self.range_geometry) in _get_identity_geometries():
            raise NotImplementedError("Gradient not implemented for model {} with range geometry {}".format(self,self.range_geometry)) 
        
        # Raise an error if wrt is passed as function value and the domain_geometry 
        # does not have fun2par method
        error_message = \
         "For the gradient to be computed, is_wrt_par "+\
         "needs to be True and wrt needs to be parameter value, not function "+\
         f"value. Alternatively, the model domain_geometry: {self.domain_geometry} "+\
         "should have an implementation of the method fun2par"
        if isinstance(wrt, CUQIarray):
            wrt_par = wrt.parameters
        elif is_wrt_par:
            wrt_par = wrt
        else:
            try:
                wrt_par = self.domain_geometry.fun2par(wrt)
            # NotImplementedError will be raised if fun2par is not
            # implemented and ValueError will be raised when imap
            # is not set in MappedGeometry
            except NotImplementedError:
                raise NotImplementedError(error_message)
            except ValueError:
                raise ValueError(error_message + " ,including an implementation of imap for MappedGeometry")

        wrt = self._input2fun(wrt, self.domain_geometry, is_wrt_par)

        x = self._input2fun(direction,
                            self.range_geometry,
                            is_direction_par)

        grad = self._gradient_func(x, wrt)

        if hasattr(self.domain_geometry, 'gradient'):
            grad = self.domain_geometry.gradient(grad, wrt_par)

        elif type(self.domain_geometry) in _get_identity_geometries():
            grad = self._output2par(grad,
                             self.domain_geometry,
                             to_CUQIarray= (type(direction) is CUQIarray)) 

        # Raise an error if domain_geometry does not have gradient attribute and
        # is not in the list returned by `_get_identity_geometries()`. i.e. The
        # Jacobian of its par2fun map is not identity.  
        else:
            raise NotImplementedError("Gradient not implemented for model {} with domain geometry {}".format(self,self.domain_geometry))

        return grad
    
    def __len__(self):
        return self.range_dim

    def __repr__(self) -> str:
        return "CUQI {}: {} -> {}. Forward parameters: {}".format(self.__class__.__name__,self.domain_geometry,self.range_geometry,cuqi.utilities.get_non_default_args(self))
    
class LinearModel(Model):
    """Model based on a Linear forward operator.

    Parameters
    -----------
    forward : 2D ndarray or callable function.
        Forward operator.

    adjoint : 2d ndarray or callable function. (optional if matrix is passed as forward)

    range_geometry : integer or cuqi.geometry.Geometry (optional)
        If integer is given a _DefaultGeometry is created with dimension of the integer.

    domain_geometry : integer or cuqi.geometry.Geometry (optional)
        If integer is given a _DefaultGeometry is created with dimension of the integer.

    Attributes
    -----------
    range_geometry : cuqi.geometry.Geometry
        The geometry representing the range.

    domain_geometry : cuqi.geometry.Geometry
        The geometry representing the domain.

    Methods
    -----------
    :meth:`forward` the forward operator.
    :meth:`range_dim` the dimension of the range.
    :meth:`domain_dim` the dimension of the domain.
    :meth:`get_matrix` returns an ndarray with the matrix representing the forward operator.
    """
    # Linear forward model with forward and adjoint (transpose).
    
    def __init__(self,forward,adjoint=None,range_geometry=None,domain_geometry=None):
        #Assume forward is matrix if not callable (TODO: add more checks)
        if not callable(forward):      
            forward_func = lambda x: self._matrix@x
            adjoint_func = lambda y: self._matrix.T@y
            matrix = forward
        else:
            forward_func = forward
            adjoint_func = adjoint
            matrix = None

        #Check if input is callable
        if callable(adjoint_func) is not True:
            raise TypeError("Adjoint needs to be callable function of some kind")

        # Use matrix to derive range_geometry and domain_geometry
        if matrix is not None:
            if range_geometry is None:
                range_geometry = _DefaultGeometry(grid=matrix.shape[0])
            if domain_geometry is None:
                domain_geometry = _DefaultGeometry(grid=matrix.shape[1])  

        #Initialize Model class
        super().__init__(forward_func,range_geometry,domain_geometry)

        #Add adjoint
        self._adjoint_func = adjoint_func

        #Store matrix privately
        self._matrix = matrix

        #Add gradient
        self._gradient_func = lambda direction, wrt: self._adjoint_func(direction)

        # if matrix is not None: 
        #     assert(self.range_dim  == matrix.shape[0]), "The parameter 'forward' dimensions are inconsistent with the parameter 'range_geometry'"
        #     assert(self.domain_dim == matrix.shape[1]), "The parameter 'forward' dimensions are inconsistent with parameter 'domain_geometry'"

    def adjoint(self, y, is_par=True):
        """ Adjoint of the model.
        
        Adjoint converts the input to function values (if needed) using the range geometry of the model.
        Adjoint converts the output function values to parameters using the range geometry of the model.

        Parameters
        ----------
        y : ndarray or cuqi.samples.CUQIarray
            The adjoint model input.

        Returns
        -------
        ndarray or cuqi.samples.CUQIarray
            The adjoint model output. Always returned as parameters.
        """
        return self._apply_func(self._adjoint_func,
                                self.domain_geometry,
                                self.range_geometry,
                                y, is_par)


    def get_matrix(self):
        if self._matrix is not None: #Matrix exists so return it
            return self._matrix
        else:
            #TODO: Can we compute this faster while still in sparse format?
            mat = csc_matrix((self.range_dim,0)) #Sparse (m x 1 matrix)
            e = np.zeros(self.domain_dim)
            
            # Stacks sparse matrices on csc matrix
            for i in range(self.domain_dim):
                e[i] = 1
                col_vec = self.forward(e)
                mat = hstack((mat,col_vec[:,None])) #mat[:,i] = self.forward(e)
                e[i] = 0

            #Store matrix for future use
            self._matrix = mat

            return self._matrix

    def __matmul__(self, x):
        return self.forward(x)

    @property
    def T(self):
        """Transpose of linear model. Returns a new linear model acting as the transpose."""
        transpose = LinearModel(self.adjoint,self.forward,self.domain_geometry,self.range_geometry)
        if self._matrix is not None:
            transpose._matrix = self._matrix.T
        return transpose
        

class PDEModel(Model):
    """
    Model based on an underlying cuqi.pde.PDE.
    In the forward operation the PDE is assembled, solved and observed.
    
    Parameters
    -----------
    forward : 2D ndarray or callable function.
        Forward operator assembling, solving and observing the pde.

    range_geometry : integer or cuqi.geometry.Geometry (optional)
        If integer is given a _DefaultGeometry is created with dimension of the integer.

    domain_geometry : integer or cuqi.geometry.Geometry (optional)
        If integer is given a _DefaultGeometry is created with dimension of the integer.

    Attributes
    -----------
    range_geometry : cuqi.geometry.Geometry
        The geometry representing the range.

    domain_geometry : cuqi.geometry.Geometry
        The geometry representing the domain.

    Methods
    -----------
    :meth:`forward` the forward operator.
    :meth:`range_dim` the dimension of the range.
    :meth:`domain_dim` the dimension of the domain.
    """
    def __init__(self, PDE, range_geometry, domain_geometry):
        #....
        if not isinstance(PDE,cuqi.pde.PDE):
            raise ValueError("PDE needs to be a cuqi PDE.")

        super().__init__(self._forward_func, range_geometry, domain_geometry)

        self.pde = PDE
        if hasattr(self.pde, "gradient_wrt_parameter"):
            self._gradient_func = self.pde.gradient_wrt_parameter

    def _forward_func(self,x):
        
        self.pde.assemble(parameter=x)

        sol, info = self.pde.solve()

        obs = self.pde.observe(sol)

        return obs

    # Add the underlying PDE class name to the repr.
    def __repr__(self) -> str:
        return super().__repr__()+". PDE: {}".format(self.pde.__class__.__name__)
        
