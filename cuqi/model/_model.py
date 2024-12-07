from __future__ import annotations
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse import hstack
from scipy.linalg import solve
from cuqi.samples import Samples
from cuqi.array import CUQIarray
from cuqi.geometry import Geometry, _DefaultGeometry1D, _DefaultGeometry2D, _get_identity_geometries
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
        If integer is given, a cuqi.geometry._DefaultGeometry is created with dimension of the integer.

    domain_geometry : integer or cuqi.geometry.Geometry
        If integer is given, a cuqi.geometry._DefaultGeometry is created with dimension of the integer.

    gradient : callable function, optional
        The direction-Jacobian product of the forward operator Jacobian with 
        respect to the forward operator input, evaluated at a point (`wrt`).
        The signature of the gradient function should be (`direction`, `wrt`),
        where `direction` is the direction by which the Jacobian matrix is
        multiplied and `wrt` is the point at which the Jacobian is computed.

    jacobian : callable function, optional
        The Jacobian of the forward operator with respect to the forward operator input,
        evaluated at a point (`wrt`). The signature of the Jacobian function should be (`wrt`).
        The Jacobian function should return a 2D ndarray of shape (range_dim, domain_dim).
        The Jacobian function is used to specify the gradient function by computing the vector-Jacobian
        product (VJP), here we refer to the vector in the VJP as the `direction` since it is the direction at 
        which the gradient is computed.
        automatically and thus the gradient function should not be specified when the Jacobian
        function is specified.


    :ivar range_geometry: The geometry representing the range.
    :ivar domain_geometry: The geometry representing the domain.

    Example
    -------

    Consider a forward model :math:`F: \mathbb{R}^2 \\rightarrow \mathbb{R}` defined by the following forward operator:

    .. math::

        F(x) = 10x_2 - 10x_1^3 + 5x_1^2 + 6x_1

    The jacobian matrix of the forward operator is given by:

    .. math::

        J_F(x) = \\begin{bmatrix} -30x_1^2 + 10x_1 + 6 & 10 \\end{bmatrix}

    The forward model can be defined as follows:

    .. code-block:: python

        import numpy as np
        from cuqi.model import Model

        def forward(x):
            return 10*x[1] - 10*x[0]**3 + 5*x[0]**2 + 6*x[0]

        def jacobian(x): # Can use "x" or "wrt" as the input argument name
            return np.array([[-30*x[0]**2 + 10*x[0] + 6, 10]])

        model = Model(forward, range_geometry=1, domain_geometry=2, jacobian=jacobian)

    Alternatively, the gradient information in the forward model can be defined by direction-Jacobian product using the gradient keyword argument.

    This may be more efficient if forming the Jacobian matrix is expensive.

    .. code-block:: python

        import numpy as np
        from cuqi.model import Model

        def forward(x):
            return 10*x[1] - 10*x[0]**3 + 5*x[0]**2 + 6*x[0]

        def gradient(direction, wrt):
            # Direction-Jacobian product direction@jacobian(wrt)
            return direction@np.array([[-30*wrt[0]**2 + 10*wrt[0] + 6, 10]])

        model = Model(forward, range_geometry=1, domain_geometry=2, gradient=gradient)

    """
    def __init__(self, forward, range_geometry, domain_geometry, gradient=None, jacobian=None):

        #Check if input is callable
        if callable(forward) is not True:
            raise TypeError("Forward needs to be callable function.")
        
        # Check if only one of gradient and jacobian is given
        if (gradient is not None) and (jacobian is not None):
            raise TypeError("Only one of gradient and jacobian should be specified")
        
        #Check if input is callable
        if (gradient is not None) and (callable(gradient) is not True):
            raise TypeError("Gradient needs to be callable function.")
        
        if (jacobian is not None) and (callable(jacobian) is not True):
            raise TypeError("Jacobian needs to be callable function.")
        
        # Use jacobian function to specify gradient function (vector-Jacobian product)
        if jacobian is not None:
            gradient = lambda direction, wrt: direction@jacobian(wrt)
 
        #Store forward func
        self._forward_func = forward
        self._gradient_func = gradient
         
        #Store range_geometry
        if isinstance(range_geometry, tuple) and len(range_geometry) == 2:
            self.range_geometry = _DefaultGeometry2D(range_geometry)
        elif isinstance(range_geometry, int):
            self.range_geometry = _DefaultGeometry1D(grid=range_geometry)
        elif isinstance(range_geometry, Geometry):
            self.range_geometry = range_geometry
        elif range_geometry is None:
            raise AttributeError("The parameter 'range_geometry' is not specified by the user and it connot be inferred from the attribute 'forward'.")
        else:
            raise TypeError("The parameter 'range_geometry' should be of type 'int', 2 dimensional 'tuple' or 'cuqi.geometry.Geometry'.")

        #Store domain_geometry
        if isinstance(domain_geometry, tuple) and len(domain_geometry) == 2:
            self.domain_geometry = _DefaultGeometry2D(domain_geometry)
        elif isinstance(domain_geometry, int):
            self.domain_geometry = _DefaultGeometry1D(grid=domain_geometry)
        elif isinstance(domain_geometry, Geometry):
            self.domain_geometry = domain_geometry
        elif domain_geometry is None:
            raise AttributeError("The parameter 'domain_geometry' is not specified by the user and it connot be inferred from the attribute 'forward'.")
        else:
            raise TypeError("The parameter 'domain_geometry' should be of type 'int', 2 dimensional 'tuple' or 'cuqi.geometry.Geometry'.")

        # Store non_default_args of the forward operator for faster caching when checking for those arguments.
        self._non_default_args = cuqi.utilities.get_non_default_args(self._forward_func)

    @property
    def domain_dim(self):
        """
        The dimension of the domain
        """
        return self.domain_geometry.par_dim

    @property
    def range_dim(self): 
        """
        The dimension of the range
        """
        return self.range_geometry.par_dim

    def _2fun(self, x, geometry, is_par):
        """ Converts `x` to function values (if needed) using the appropriate 
        geometry. For example, `x` can be the model input which need to be
        converted to function value before being passed to 
        :class:`~cuqi.model.Model` operators (e.g. _forward_func, _adjoint_func,
        _gradient_func).

        Parameters
        ----------
        x : ndarray or cuqi.array.CUQIarray
            The value to be converted.

        geometry : cuqi.geometry.Geometry
            The geometry representing `x`.

        is_par : bool
            If True, `x` is assumed to be parameters.
            If False, `x` is assumed to be function values.

        Returns
        -------
        ndarray or cuqi.array.CUQIarray
            `x` represented as a function.
        """
        # Convert to function representation
        # if x is CUQIarray and geometry are consistent, we obtain funvals
        # directly
        if isinstance(x, CUQIarray) and  x.geometry == geometry:
            x = x.funvals
        # Otherwise we use the geometry par2fun method
        elif is_par:
            x = geometry.par2fun(x)

        return x

    def _2par(self, val, geometry, to_CUQIarray=False, is_par=False):
        """ Converts val, normally output of :class:~`cuqi.model.Model` 
        operators (e.g. _forward_func, _adjoint_func, _gradient_func), to
        parameters using the appropriate geometry.

        Parameters
        ----------
        val : ndarray or cuqi.array.CUQIarray
            The value to be converted to parameters.

        geometry : cuqi.geometry.Geometry
            The geometry representing the argument `val`.

        to_CUQIarray : bool
            If True, the returned value is wrapped as a cuqi.array.CUQIarray.
        
        is_par : bool
            If True, `val` is assumed to be of parameter representation and
            hence no conversion to parameters is performed.

        Returns
        -------
        ndarray or cuqi.array.CUQIarray
            The value `val` represented as parameters.
        """
        # Convert to parameters
        # if val is CUQIarray and geometry are consistent, we obtain parameters
        # directly
        if isinstance(val, CUQIarray) and val.geometry == geometry:
            val = val.parameters
        # Otherwise we use the geometry fun2par method
        elif not is_par:
            val = geometry.fun2par(val)

        # Wrap val in CUQIarray if requested
        if to_CUQIarray:
            val = CUQIarray(val, is_par=True, geometry=geometry)

        # Return val
        return val
        

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

        x : ndarray or cuqi.array.CUQIarray
            The input value to the operator.

        is_par : bool
            If True the input is assumed to be parameters.
            If False the input is assumed to be function values.

        Returns
        -------
        ndarray or cuqi.array.CUQIarray
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
        
        # store if input x is CUQIarray
        is_CUQIarray = type(x) is CUQIarray

        x = self._2fun(x, func_domain_geometry, is_par=is_par)
        out = func(x, **kwargs)

        # Return output as parameters 
        # (and wrapped in CUQIarray if input was CUQIarray)
        return self._2par(out, func_range_geometry, 
                                    to_CUQIarray=is_CUQIarray)

    def _parse_args_add_to_kwargs(self, *args, **kwargs):
        """ Private function that parses the input arguments of the model and adds them as keyword arguments matching the non default arguments of the forward function. """

        if len(args) > 0:

            if len(kwargs) > 0:
                raise ValueError("The model input is specified both as positional and keyword arguments. This is not supported.")
                
            if len(args) != len(self._non_default_args):
                raise ValueError("The number of positional arguments does not match the number of non-default arguments of the model.")
            
            # Add args to kwargs following the order of non_default_args
            for idx, arg in enumerate(args):
                kwargs[self._non_default_args[idx]] = arg

        return kwargs
        
    def forward(self, *args, is_par=True, **kwargs):
        """ Forward function of the model.
        
        Forward converts the input to function values (if needed) using the domain geometry of the model.
        Forward converts the output function values to parameters using the range geometry of the model.

        Parameters
        ----------
        *args : ndarray or cuqi.array.CUQIarray
            The model input.

        is_par : bool
            If True the input is assumed to be parameters.
            If False the input is assumed to be function values.
        
        **kwargs : keyword arguments for model input.
            Keywords must match the names of the non_default_args of the model.

        Returns
        -------
        ndarray or cuqi.array.CUQIarray
            The model output. Always returned as parameters.
        """

        kwargs = self._parse_args_add_to_kwargs(*args, **kwargs)

        # Check kwargs matches non_default_args
        if set(list(kwargs.keys())) != set(self._non_default_args):
            raise ValueError(f"The model input is specified by a keywords arguments {kwargs.keys()} that does not match the non_default_args of the model {self._non_default_args}.")

        # For now only support one input
        if len(kwargs) > 1:
            raise ValueError("The model input is specified by more than one argument. This is not supported.")

        # Get input matching the non_default_args
        x = kwargs[self._non_default_args[0]]

        # If input is a distribution, we simply change the parameter name of model to match the distribution name
        if isinstance(x, cuqi.distribution.Distribution):
            if x.dim != self.domain_dim:
                raise ValueError("Attempting to match parameter name of Model with given distribution, but distribution dimension does not match model domain dimension.")
            new_model = copy(self)
            new_model._non_default_args = [x.name] # Defaults to x if distribution had no name
            return new_model

        # If input is a random variable, we handle it separately
        if isinstance(x, cuqi.experimental.algebra.RandomVariable):
            return self._handle_random_variable(x)
        
        # If input is a Node from internal abstract syntax tree, we let the Node handle the operation
        # We use NotImplemented to indicate that the operation is not supported from the Model class
        # in case of operations such as "@" that can be interpreted as both __matmul__ and __rmatmul__
        # the operation may be delegated to the Node class.
        if isinstance(x, cuqi.experimental.algebra.Node):
            return NotImplemented

        # Else we apply the forward operator
        return self._apply_func(self._forward_func,
                                self.range_geometry,
                                self.domain_geometry,
                                x, is_par)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def gradient(self, direction, wrt, is_direction_par=True, is_wrt_par=True):
        """ Gradient of the forward operator (Direction-Jacobian product)

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
        # Obtain the parameters representation of wrt and raise an error if it
        # cannot be obtained
        error_message = \
            "For the gradient to be computed, is_wrt_par needs " +\
            "to be True and wrt needs to be parameter value, not function " +\
            "value. Alternatively, the model domain_geometry: "+\
            f"{self.domain_geometry} " +\
            "should have an implementation of the method fun2par"
        try:
            wrt_par = self._2par(wrt, 
                                 geometry=self.domain_geometry,
                                 is_par=is_wrt_par,
                                 to_CUQIarray=False,
                                 )
        # NotImplementedError will be raised if fun2par of the geometry is not
        # implemented and ValueError will be raised when imap is not set in
        # MappedGeometry
        except ValueError as e:
            raise ValueError(error_message +
                             " ,including an implementation of imap for " +
                             "MappedGeometry")
        except NotImplementedError as e:
            raise NotImplementedError(error_message)
        
        # Check for other errors that may prevent computing the gradient
        self._check_gradient_can_be_computed(direction, wrt)

        wrt = self._2fun(wrt, self.domain_geometry, is_par=is_wrt_par)

        # Store if the input direction is CUQIarray
        is_direction_CUQIarray = type(direction) is CUQIarray

        direction = self._2fun(direction,
                               self.range_geometry,
                               is_par=is_direction_par)

        grad = self._gradient_func(direction, wrt)
        grad_is_par = False # Assume gradient is function values
        
        # If domain_geometry has gradient attribute, we apply it to the gradient
        # The gradient returned by the domain_geometry.gradient is assumed to be
        # parameters
        if hasattr(self.domain_geometry, 'gradient'):
            grad = self.domain_geometry.gradient(grad, wrt_par)
            grad_is_par = True # Gradient is parameters

        # we convert the computed gradient to parameters
        grad = self._2par(grad,
                          self.domain_geometry,
                          to_CUQIarray=is_direction_CUQIarray,
                          is_par=grad_is_par)

        return grad
    
    def _check_gradient_can_be_computed(self, direction, wrt):
        """ Private function that checks if the gradient can be computed. By
        raising an error for the cases where the gradient cannot be computed."""

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
        
        # Raise an error if domain_geometry does not have gradient attribute and
        # is not in the list returned by `_get_identity_geometries()`. i.e. the
        # Jacobian of its par2fun map is not identity.  
        if not hasattr(self.domain_geometry, 'gradient') and \
            not type(self.domain_geometry) in _get_identity_geometries():
            raise NotImplementedError("Gradient not implemented for model {} with domain geometry {}".format(self,self.domain_geometry))

    def _handle_random_variable(self, x):
        """ Private function that handles the case of the input being a random variable. """
        # If random variable is not a leaf-type node (e.g. internal node) we return NotImplemented
        if not isinstance(x.tree, cuqi.experimental.algebra.VariableNode):
            return NotImplemented        
        
        # In leaf-type node case we simply change the parameter name of model to match the random variable name
        dist = x.distribution
        if dist.dim != self.domain_dim:
            raise ValueError("Attempting to match parameter name of Model with given random variable, but random variable dimension does not match model domain dimension.")
        
        new_model = copy(self)
        new_model._non_default_args = [dist.name]
        return new_model

    def __len__(self):
        return self.range_dim

    def __repr__(self) -> str:
        return "CUQI {}: {} -> {}.\n    Forward parameters: {}.".format(self.__class__.__name__,self.domain_geometry,self.range_geometry,cuqi.utilities.get_non_default_args(self))


class AffineModel(Model):
    """ Model class representing an affine model, i.e. a linear operator with a fixed shift. For linear models, represented by a linear operator only, see :class:`~cuqi.model.LinearModel`.

    The affine model is defined as:

    .. math::

        x \\mapsto Ax + shift

    where :math:`A` is the linear operator and :math:`shift` is the shift.

    Parameters
    ----------

    linear_operator : 2d ndarray, callable function or cuqi.model.LinearModel
        The linear operator. If ndarray is given, the operator is assumed to be a matrix.

    shift : scalar or array_like
        The shift to be added to the forward operator.

    linear_operator_adjoint : callable function, optional
        The adjoint of the linear operator. Also used for computing gradients.

    range_geometry : cuqi.geometry.Geometry
        The geometry representing the range.

    domain_geometry : cuqi.geometry.Geometry
        The geometry representing the domain.

    """

    def __init__(self, linear_operator, shift, linear_operator_adjoint=None, range_geometry=None, domain_geometry=None):

        # If input represents a matrix, extract needed properties from it
        if hasattr(linear_operator, '__matmul__') and hasattr(linear_operator, 'T'):
            if linear_operator_adjoint is not None:
                raise ValueError("Adjoint of linear operator should not be provided when linear operator is a matrix. If you want to provide an adjoint, use a callable function for the linear operator.")
            
            matrix = linear_operator

            linear_operator = lambda x: matrix@x
            linear_operator_adjoint = lambda y: matrix.T@y

            if range_geometry is None:
                if hasattr(matrix, 'shape'):
                    range_geometry = _DefaultGeometry1D(grid=matrix.shape[0])
                elif isinstance(matrix, LinearModel):
                    range_geometry = matrix.range_geometry

            if domain_geometry is None:
                if hasattr(matrix, 'shape'):
                    domain_geometry = _DefaultGeometry1D(grid=matrix.shape[1])
                elif isinstance(matrix, LinearModel):
                    domain_geometry = matrix.domain_geometry
        else:
            matrix = None

        # Ensure that the operators are a callable functions (either provided or created from matrix)
        if not callable(linear_operator):
            raise TypeError("Linear operator must be defined as a matrix or a callable function of some kind")
        if linear_operator_adjoint is not None and not callable(linear_operator_adjoint):
            raise TypeError("Linear operator adjoint must be defined as a callable function of some kind")

        # Check size of shift and match against range_geometry
        if not np.isscalar(shift):
            if len(shift) != range_geometry.par_dim:
                raise ValueError("The shift should have the same dimension as the range geometry.")

        # Initialize Model class
        super().__init__(linear_operator, range_geometry, domain_geometry)

        # Store matrix privately
        self._matrix = matrix

        # Store shift as private attribute
        self._shift = shift

        # Store linear operator privately
        self._linear_operator = linear_operator

        # Store adjoint function
        self._linear_operator_adjoint = linear_operator_adjoint

        # Define gradient
        self._gradient_func = lambda direction, wrt: linear_operator_adjoint(direction)

        # Update forward function to include shift (overwriting the one from Model class)
        self._forward_func = lambda *args, **kwargs: linear_operator(*args, **kwargs) + shift

        # Use arguments from user's callable linear operator (overwriting those found by Model class)
        self._non_default_args = cuqi.utilities.get_non_default_args(linear_operator)

    @property
    def shift(self):
        """ The shift of the affine model. """
        return self._shift

    @shift.setter
    def shift(self, value):
        """ Update the shift of the affine model. Updates both the shift value and the underlying forward function. """
        self._shift = value
        self._forward_func = lambda *args, **kwargs: self._linear_operator(*args, **kwargs) + value

    def _forward_func_no_shift(self, x, is_par=True):
        """ Helper function for computing the forward operator without the shift. """
        return self._apply_func(self._linear_operator,
                self.range_geometry,
                self.domain_geometry,
                x, is_par)

    def _adjoint_func_no_shift(self, y, is_par=True):
        """ Helper function for computing the adjoint operator without the shift. """
        return self._apply_func(self._linear_operator_adjoint,
                self.domain_geometry,
                self.range_geometry,
                y, is_par)

class LinearModel(AffineModel):
    """Model based on a Linear forward operator.

    Parameters
    -----------
    forward : 2D ndarray or callable function.
        Forward operator.

    adjoint : 2D ndarray or callable function. (optional if matrix is passed as forward)

    range_geometry : integer or cuqi.geometry.Geometry (optional)
        If integer is given, a cuqi.geometry._DefaultGeometry is created with dimension of the integer.

    domain_geometry : integer or cuqi.geometry.Geometry (optional)
        If integer is given, a cuqi.geometry._DefaultGeometry is created with dimension of the integer.


    :ivar range_geometry: The geometry representing the range.
    :ivar domain_geometry: The geometry representing the domain.

    Example
    -------

    Consider a linear model represented by a matrix, i.e., :math:`y=Ax` where 
    :math:`A` is a matrix.
    
    We can define such a linear model by passing the matrix :math:`A`:

    .. code-block:: python

        import numpy as np
        from cuqi.model import LinearModel

        A = np.random.randn(2,3)

        model = LinearModel(A)

    The dimension of the range and domain geometries will be automatically 
    inferred from the matrix :math:`A`.
        
    Meanwhile, such a linear model can also be defined by a forward function 
    and an adjoint function:

    .. code-block:: python

        import numpy as np
        from cuqi.model import LinearModel

        A = np.random.randn(2,3)

        def forward(x):
            return A@x

        def adjoint(y):
            return A.T@y

        model = LinearModel(forward,
                            adjoint=adjoint,
                            range_geometry=2,
                            domain_geometry=3)

    Note that you would need to specify the range and domain geometries in this
    case as they cannot be inferred from the forward and adjoint functions.
    """
    
    def __init__(self, forward, adjoint=None, range_geometry=None, domain_geometry=None):

        #Initialize as AffineModel with shift=0
        super().__init__(forward, 0, adjoint, range_geometry, domain_geometry)

    def adjoint(self, y, is_par=True):
        """ Adjoint of the model.
        
        Adjoint converts the input to function values (if needed) using the range geometry of the model.
        Adjoint converts the output function values to parameters using the range geometry of the model.

        Parameters
        ----------
        y : ndarray or cuqi.array.CUQIarray
            The adjoint model input.

        Returns
        -------
        ndarray or cuqi.array.CUQIarray
            The adjoint model output. Always returned as parameters.
        """
        if self._linear_operator_adjoint is None:
            raise ValueError("No adjoint operator was provided for this model.")
        return self._apply_func(self._linear_operator_adjoint,
                                self.domain_geometry,
                                self.range_geometry,
                                y, is_par)

    def __matmul__(self, x):
        return self.forward(x)
        
    def get_matrix(self):
        """
        Returns an ndarray with the matrix representing the forward operator.
        """

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

    @property
    def T(self):
        """Transpose of linear model. Returns a new linear model acting as the transpose."""
        transpose = LinearModel(self.adjoint, self.forward, self.domain_geometry, self.range_geometry)
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
        If integer is given, a cuqi.geometry._DefaultGeometry is created with dimension of the integer.

    domain_geometry : integer or cuqi.geometry.Geometry (optional)
        If integer is given, a cuqi.geometry._DefaultGeometry is created with dimension of the integer.


    :ivar range_geometry: The geometry representing the range.
    :ivar domain_geometry: The geometry representing the domain.
    """
    def __init__(self, PDE: cuqi.pde.PDE, range_geometry, domain_geometry):

        if not isinstance(PDE, cuqi.pde.PDE):
            raise ValueError("PDE needs to be a cuqi PDE.")

        super().__init__(self._forward_func, range_geometry, domain_geometry, gradient=self._gradient_func)

        self.pde = PDE

    def _forward_func(self, x):
        
        self.pde.assemble(parameter=x)

        sol, info = self.pde.solve()

        obs = self.pde.observe(sol)

        return obs
    
    def _gradient_func(self, direction, wrt):
        """ Compute direction-Jacobian product (gradient) of the model. """
        if hasattr(self.pde, "gradient_wrt_parameter"):
            return self.pde.gradient_wrt_parameter(direction, wrt)
        elif hasattr(self.pde, "jacobian_wrt_parameter"):
            return direction@self.pde.jacobian_wrt_parameter(wrt)
        else:
            raise NotImplementedError("Gradient is not implemented for this model.")

    # Add the underlying PDE class name to the repr.
    def __repr__(self) -> str:
        return super().__repr__()+"\n    PDE: {}.".format(self.pde.__class__.__name__)
        
