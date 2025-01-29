from __future__ import annotations
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse import hstack
from scipy.linalg import solve
from cuqi.samples import Samples
from cuqi.array import CUQIarray
from cuqi.geometry import Geometry, _DefaultGeometry1D, _DefaultGeometry2D,\
    _get_identity_geometries
import cuqi
import matplotlib.pyplot as plt
from copy import copy
from functools import partial

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

        # Check if input is callable
        if callable(forward) is not True:
            raise TypeError("Forward needs to be callable function.")

        # Store forward func
        self._forward_func = forward
        self._stored_non_default_args = None

        # Check if only one of gradient and jacobian is given
        if (gradient is not None) and (jacobian is not None):
            raise TypeError("Only one of gradient and jacobian should be specified")

        # Check correct gradient form (check type, signature, etc.)
        self._check_correct_gradient_jacobian_form(gradient, "gradient")

        # Check correct jacobian form (check type, signature, etc.)
        self._check_correct_gradient_jacobian_form(jacobian, "jacobian")

        # If jacobian is provided, use it to specify gradient function
        # (vector-Jacobian product)
        if jacobian is not None:
            gradient = self._use_jacobian_to_specify_gradient(jacobian)

        self._gradient_func = gradient

        # Store range_geometry
        self.range_geometry = range_geometry

        # Store domain_geometry
        self.domain_geometry = domain_geometry

    @property
    def _non_default_args(self):
        if self._stored_non_default_args is None:
            # Store non_default_args of the forward operator for faster caching
            # when checking for those arguments.
            self._stored_non_default_args =\
                cuqi.utilities.get_non_default_args(self._forward_func)
        return self._stored_non_default_args

    @property
    def range_geometry(self):
        """ The geometry representing the range of the model. """
        return self._range_geometry

    @range_geometry.setter
    def range_geometry(self, value):
        """ Update the range geometry of the model. """
        if isinstance(value, tuple) and len(value) == 2:
            self._range_geometry = _DefaultGeometry2D(value)
        elif isinstance(value, int):
            self._range_geometry = _DefaultGeometry1D(grid=value)
        elif isinstance(value, Geometry):
            self._range_geometry = value
        elif value is None:
            raise AttributeError(
                "The parameter 'range_geometry' is not specified by the user and it cannot be inferred from the attribute 'forward'."
            )
        else:
            raise TypeError(
                "The parameter 'range_geometry' should be of type 'int', 2 dimensional 'tuple' or 'cuqi.geometry.Geometry'."
            )

    @property
    def domain_geometry(self):
        """ The geometry representing the domain of the model. """
        return self._domain_geometry

    @domain_geometry.setter
    def domain_geometry(self, value):
        """ Update the domain geometry of the model. """
        # Forward model with multiple input case
        if isinstance(value, tuple) and\
            all([isinstance(v, Geometry) for v in value]):
            self._domain_geometry = cuqi.experimental.geometry._ProductGeometry(
                *value)
        # Remaining cases, forward model with single input
        elif isinstance(value, tuple) and len(value) == 2:
            self._domain_geometry = _DefaultGeometry2D(value)
        elif isinstance(value, int):
            self._domain_geometry = _DefaultGeometry1D(grid=value)
        elif isinstance(value, Geometry):
            self._domain_geometry = value
        elif value is None:
            raise AttributeError(
                "The parameter 'domain_geometry' is not specified by the user and it cannot be inferred from the attribute 'forward'."
            )
        else:
            raise TypeError(
                "The parameter 'domain_geometry' should be of type 'int', 2 dimensional 'tuple' or 'cuqi.geometry.Geometry'."
            )

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

    def _check_correct_gradient_jacobian_form(self, func, func_type):
        """Private function that checks if the gradient/jacobian parameter is
        in the correct form. That is, check if the gradient/jacobian has the
        correct type, signature, etc."""

        if func is None:
            return

        # gradient/jacobian should be callable (for single input and multiple input case)
        # or a tuple of callables (for multiple inputs case)
        if isinstance(func, tuple):
            # tuple length should be same as the number of inputs
            if len(func) != len(self._non_default_args):
                raise ValueError(
                    f"The "
                    + func_type.lowe()
                    + " tuple length should be {len(self._non_default_args)} for model with inputs {self._non_default_args}"
                )
            # tuple items should be callables or None
            if not all([callable(func_i) or func_i is None for func_i in func]):
                raise TypeError(
                    func_type.capitalize()
                    + " tuple should contain callable functions or None."
                )

        elif callable(func):
            # temporarily convert gradient/jacobian to tuple for checking only
            func = (func,)

        else:
            raise TypeError(
                "Gradient needs to be callable function or tuple of callable functions."
            )

        expected_func_non_default_args = self._non_default_args
        if func_type.lower() == "gradient":
            # prepend 'direction' to the expected gradient non default args
            expected_func_non_default_args = [
                "direction"
            ] + expected_func_non_default_args

        for func_i in func:
            # make sure the signature of the gradient/jacobian function is correct
            # that is, the same as the expected_func_non_default_args
            if func_i is not None:
                func_non_default_args = cuqi.utilities.get_non_default_args(func_i)

                if set(func_non_default_args) != set(expected_func_non_default_args):
                    raise ValueError(
                        func_type.capitalize()
                        + f" function signature should be {expected_func_non_default_args}"
                    )

    def _use_jacobian_to_specify_gradient(self, jacobian):
        """Private function that uses the jacobian function to specify the
        gradient function."""
        # if jacobian is a single function
        if callable(jacobian):
            gradient = self._create_gradient_lambda_function_from_jacobian_with_correct_signature(
                jacobian
            )

        # Else, jacobian is a tuple of jacobian functions
        else:
            gradient = []
            for jac in jacobian:
                if jac is not None:
                    gradient.append(
                        self._create_gradient_lambda_function_from_jacobian_with_correct_signature(
                            jac
                        )
                    )
                else:
                    gradient.append(None)
        return tuple(gradient) if isinstance(gradient, list) else gradient

    def _create_gradient_lambda_function_from_jacobian_with_correct_signature(
        self, jacobian
    ):
        """Private function that creates gradient lambda function from the
        jacobian function, with the correct signature (based on the model
        non_default_args).
        """
        # create the string representation of the lambda function
        grad_fun_str = (
            "lambda direction, "
            + ", ".join(self._non_default_args)
            + ", jacobian: direction@jacobian("
            + ", ".join(self._non_default_args)
            + ")"
        )

        # create the lambda function from the string
        grad_func = eval(grad_fun_str)

        # create partial function from the lambda function with jacobian as a
        # fixed argument
        grad_func = partial(grad_func, jacobian=jacobian)

        return grad_func

    def _2fun(self, geometry=None, is_par=True, **kwargs):
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
        # Check kwargs and geometry are consistent and set up geometries list and
        # is_par tuple
        geometries, is_par = self._helper_pre_conversion_checks_and_processing(geometry, is_par, **kwargs)

        # Convert to function values
        for i, (k, v) in enumerate(kwargs.items()):
            # Use CUQIarray funvals if geometry is consistent
            if isinstance(v, CUQIarray) and v.geometry == geometries[i]:
                kwargs[k] = v.funvals
            # Otherwise we use the geometry par2fun method
            elif is_par[i] and v is not None:
                kwargs[k] = geometries[i].par2fun(v)

        return kwargs

    def _helper_pre_conversion_checks_and_processing(self, geometry=None, is_par=True, **kwargs):
        """ Helper function that checks if kwargs and geometry are consistent
        and sets up geometries list and is_par tuple.
        """
        # If len of kwargs is larger than 1, the geometry needs to be of type
        # _ProductGeometry
        if (
            not isinstance(geometry, cuqi.experimental.geometry._ProductGeometry)
            and len(kwargs) > 1
        ):
            raise ValueError(
                "The input is specified by more than one argument. This is only "
                + "supported for domain geometry of type "
                + f"{cuqi.experimental.geometry._ProductGeometry.__name__}."
            )

        # If is_par is bool, make it a tuple of bools of the same length as
        # kwargs
        is_par = (is_par,) * len(kwargs) if isinstance(is_par, bool) else is_par

        # Set up geometries list
        geometries = (
            geometry.geometries
            if isinstance(geometry, cuqi.experimental.geometry._ProductGeometry)
            else [geometry]
        )

        return geometries, is_par

    def _2par(self, geometry=None, to_CUQIarray=False, is_par=False, **kwargs):    
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
        # Check kwargs and geometry are consistent and set up geometries list and
        # is_par tuple
        geometries, is_par = self._helper_pre_conversion_checks_and_processing(geometry, is_par, **kwargs)

        # if to_CUQIarray is bool, make it a tuple of bools of the same length
        # as kwargs
        to_CUQIarray = (to_CUQIarray,) * len(kwargs) if isinstance(to_CUQIarray, bool) else to_CUQIarray

        # Convert to parameters
        for i , (k, v) in enumerate(kwargs.items()):
            # Use CUQIarray parameters if geometry is consistent
            if isinstance(v, CUQIarray) and v.geometry == geometries[i]:
                v = v.parameters
            # Otherwise we use the geometry fun2par method
            elif not is_par[i] and v is not None:
                v = geometries[i].fun2par(v)

            # Wrap the value v in CUQIarray if requested
            if to_CUQIarray[i] and v is not None:
                v = CUQIarray(v, is_par=True, geometry=geometries[i])

            kwargs[k] = v

        return kwargs

    def _apply_func(self, func=None, fwd=True, is_par=True, **kwargs):
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
        # Specify the range and domain geometries of the function
        # If forward operator, range geometry is the model range geometry and
        # domain geometry is the model domain geometry
        if fwd:
            func_range_geometry = self.range_geometry
            func_domain_geometry = self.domain_geometry
        # If adjoint operator, range geometry is the model domain geometry and
        # domain geometry is the model range geometry
        else:
            func_range_geometry = self.domain_geometry
            func_domain_geometry = self.range_geometry

        # If input x is Samples we apply func for each sample
        # TODO: Check if this can be done all-at-once for computational speed-up
        if any(isinstance(x, Samples) for x in kwargs.values()):
            return self._handle_samples(func, fwd, is_par, **kwargs)

        # store if any input x is CUQIarray
        is_CUQIarray = any(isinstance(x, CUQIarray) for x in kwargs.values())

        # Convert input to function values
        kwargs = self._2fun(geometry=func_domain_geometry, is_par=is_par, **kwargs)

        # Apply the function
        out = func(**kwargs)

        # Return output as parameters
        # (wrapped in CUQIarray if any input was CUQIarray)
        return self._2par(
            geometry=func_range_geometry, to_CUQIarray=is_CUQIarray, **{"out": out}
        )["out"]

    def _handle_samples(self, func=None, fwd=True, is_par=True, **kwargs):
        """Private function that calls apply_func for each sample in the
        Samples object.
        """
        # All kwargs should be Samples objects
        if not all(isinstance(x, Samples) for x in kwargs.values()):
            raise TypeError(
                "If applying the function to Samples, all inputs should be Samples."
            )

        # All Samples objects should have the same number of samples
        Ns = list(kwargs.values())[0].Ns
        if not all(x.Ns == Ns for x in kwargs.values()):
            raise ValueError(
                "If applying the function to Samples, all inputs should have the same number of samples."
            )

        # Specify the range dimension of the function
        range_dim = self.range_dim if fwd else self.domain_dim

        # Create empty array to store the output
        out = np.zeros((range_dim, Ns))

        # Recursively apply func to each sample
        for i in range(Ns):
            kwargs_i = {k: v.samples[..., i] for k, v in kwargs.items()}
            out[:, i] = self._apply_func(func=func, fwd=fwd, is_par=is_par, **kwargs_i)
        # Specify the range geometries of the function
        func_range_geometry = self.range_geometry if fwd else self.domain_geometry
        return Samples(out, geometry=func_range_geometry)

    def _parse_args_add_to_kwargs(self, *args, is_par=True, non_default_args=None, **kwargs):
        """ Private function that parses the input arguments and adds them as
        keyword arguments matching (the order of) the non default arguments of
        the forward function or other specified non_default_args list.
        """
        # If non_default_args is not specified, use the non_default_args of the
        # model
        if non_default_args is None:
            non_default_args = self._non_default_args

        # If any args are given, add them to kwargs
        if len(args) > 0:
            if len(kwargs) > 0:
                raise ValueError("The model input is specified both as positional and keyword arguments. This is not supported.")

            # Check if the input is for multiple input case and is stacked,
            # then split it
            if len(args)==1 and len(non_default_args)>1:
                split_succeeded, split_args = self.is_stacked_args(*args, is_par=is_par)
                if split_succeeded:
                    args = split_args
                    appending_error_message = ""
                else:
                    appending_error_message = f" Additionally, the input is specified by a single argument that cannot be split into multiple arguments matching the non_default_args {non_default_args} of the model."

            # Check if the number of args does not match the number of
            # non_default_args of the model
            if len(args) != len(non_default_args):
                raise ValueError("The number of positional arguments does not match the number of non-default arguments of the model."+appending_error_message)

            # Add args to kwargs following the order of non_default_args
            for idx, arg in enumerate(args):
                kwargs[non_default_args[idx]] = arg

        # Check kwargs matches non_default_args
        if set(list(kwargs.keys())) != set(non_default_args):
            raise ValueError(f"The model input is specified by a keywords arguments {kwargs.keys()} that does not match the non_default_args of the model {non_default_args}.")

        # Make sure order of kwargs is the same as non_default_args
        kwargs = {k: kwargs[k] for k in non_default_args}

        return kwargs

    def _parse_args_add_to_kwargs_forward(self, *args, is_par=True, **kwargs):
        """ Private function that parses the input arguments of the model and 
        adds them as keyword arguments matching the (order of) non default arguments of
        the forward function. """
        kwargs = self._parse_args_add_to_kwargs(
            *args, is_par=is_par, non_default_args=None, **kwargs)

        # For now only support either the case of one input or the case of
        # multiple inputs with _ProductGeometry as the domain_geometry
        if not isinstance(self.domain_geometry, cuqi.experimental.geometry._ProductGeometry)\
            and len(kwargs) > 1:
            raise ValueError(
                "The model input is specified by more than one argument. "+\
                "This is only supported for domain geometry of type "+\
                "tuple of cuqi.geometry.Geometry objects.")

        return kwargs

    def _parse_args_add_to_kwargs_gradient(self, *args, is_par=True, **kwargs):
        """ Private function that parses the input arguments of the gradient and 
        adds them as keyword arguments matching the (order of) non default arguments of
        the forward function. """
        return self._parse_args_add_to_kwargs_forward(*args, is_par=is_par, **kwargs)

    def is_stacked_args(self, *args, is_par=True):
        """Private function that checks if the input arguments are stacked
        and splits them if they are."""
        # Length of args should be 1 if the input is stacked (no partial
        # stacking is supported)
        if len(args) > 1:
            return False, args

        # Type of args should be parameter
        if not is_par:
            return False, args

        # args[0] should be numpy array or CUQIarray
        is_CUQIarray = isinstance(args[0], CUQIarray)
        is_numpy_array = isinstance(args[0], np.ndarray)
        if not is_CUQIarray and not is_numpy_array:
            return False, args

        # Shape of args[0] should be (domain_dim,)
        if not args[0].shape == (self.domain_dim,):
            return False, args

        # Ensure domain geometry is _ProductGeometry
        if not isinstance(
            self.domain_geometry, cuqi.experimental.geometry._ProductGeometry
        ):
            return False, args

        # Split the stacked input
        split_args = np.split(args[0], self.domain_geometry.stacked_par_split_indices)

        # Covert split args to CUQIarray if input is CUQIarray
        if is_CUQIarray:
            split_args = [
                CUQIarray(arg, is_par=True, geometry=self.domain_geometry.geometries[i])
                for i, arg in enumerate(split_args)
            ]

        return True, split_args

    def forward(self, *args, is_par=True, **kwargs):
        """ Forward function of the model.
        
        Forward converts the input to function values (if needed) using the domain geometry of the model.
        Forward converts the output function values to parameters using the range geometry of the model.

        Parameters
        ----------
        *args : ndarray or cuqi.array.CUQIarray
            The model input.

        is_par : bool or list of bool
            If True the input is assumed to be parameters.
            If False the input is assumed to be function values.
        
        **kwargs : keyword arguments for model input.
            Keywords must match the names of the non_default_args of the model.

        Returns
        -------
        ndarray or cuqi.array.CUQIarray
            The model output. Always returned as parameters.
        """

        # Add args to kwargs and ensure the order of the arguments matches the
        # non_default_args of the forward function
        kwargs = self._parse_args_add_to_kwargs_forward(*args, **kwargs, is_par=is_par)

        # If input is a distribution, we simply change the parameter name of
        # model to match the distribution name
        if all(isinstance(x, cuqi.distribution.Distribution)
               for x in kwargs.values()):
            return self._handle_distributions(kwargs)

        # If input is a random variable, we handle it separately
        # extract args from kwargs
        args = list(kwargs.values())
        if len(args)== 1 and isinstance(args[0], cuqi.experimental.algebra.RandomVariable):
            return self._handle_random_variable(args[0])

        # If input is a Node from internal abstract syntax tree, we let the Node handle the operation
        # We use NotImplemented to indicate that the operation is not supported from the Model class
        # in case of operations such as "@" that can be interpreted as both __matmul__ and __rmatmul__
        # the operation may be delegated to the Node class.
        if len(args)== 1 and isinstance(args[0], cuqi.experimental.algebra.Node):
            return NotImplemented

        # Else we apply the forward operator
        # if model have _original_non_default_args, we use it to replace the
        # kwargs keys so that it matches self._forward_func signature
        if hasattr(self, '_original_non_default_args'):
            kwargs = {k:v for k,v in zip(self._original_non_default_args, args)}
        return self._apply_func(func=self._forward_func,
                                fwd=True,
                                is_par=is_par,
                                **kwargs)

    def _correct_distribution_dimension(self, distributions):
        """Private function that checks if the dimension of the
        distributions matches the domain dimension of the model."""
        if len(distributions) == 1:
            return list(distributions)[0].dim == self.domain_dim
        elif len(distributions) > 1 and isinstance(
            self.domain_geometry, cuqi.experimental.geometry._ProductGeometry
        ):
            return all(
                d.dim == self.domain_geometry.par_dim_list[i]
                for i, d in enumerate(distributions)
            )
        else:
            return False

    def _handle_distributions(self, kwargs):
        """Private function that handles the case of the input being a
        distribution or multiple distributions."""

        if not self._correct_distribution_dimension(kwargs.values()):
            raise ValueError(
                "Attempting to match parameter name of Model with given distribution(s), but distribution(s) dimension(s) does not match model input dimension(s)."
            )
        new_model = copy(self)

        # Store the original non_default_args of the model
        new_model._original_non_default_args = self._non_default_args

        # Update the non_default_args of the model to match the distribution
        # names. Defaults to x in the case of one distribution that have no name
        new_model._stored_non_default_args = [x.name for x in kwargs.values()]

        # If there is a repeated name, raise an error
        if len(set(new_model._stored_non_default_args)) != len(
            new_model._stored_non_default_args
        ):
            raise ValueError(
                "Attempting to match parameter name of Model with given distributions, but distribution names are not unique. Please provide unique names for the distributions."
            )

        return new_model

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
        # Store the original non_default_args of the model
        new_model._original_non_default_args = self._non_default_args
        new_model._stored_non_default_args = [dist.name]
        return new_model

    def gradient(
        self, direction, *args, is_direction_par=True, is_var_par=True, **kwargs
    ):
        """Gradient of the forward operator (Direction-Jacobian product)

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

        is_var_par : bool
            If True, variables in kwargs are assumed to be parameters.
            If False, variables in kwargs are assumed to be function values.

        """
        # Add args to kwargs and ensure the order of the arguments matches the
        # non_default_args of the forward function
        kwargs = self._parse_args_add_to_kwargs_gradient(
            *args, **kwargs, is_par=is_var_par
        )

        # Obtain the parameters representation of the variables and raise an
        # error if it cannot be obtained
        error_message = (
            "For the gradient to be computed, is_var_par needs to be True and the variables in kwargs needs to be parameter value, not function value. Alternatively, the model domain_geometry:"
            + f" {self.domain_geometry} "
            + "should have an implementation of the method fun2par"
        )
        try:
            kwargs_par = self._2par(
                geometry=self.domain_geometry,
                is_par=is_var_par,
                to_CUQIarray=False,
                **kwargs,
            )
        # NotImplementedError will be raised if fun2par of the geometry is not
        # implemented and ValueError will be raised when imap is not set in
        # MappedGeometry
        except ValueError as e:
            raise ValueError(
                error_message
                + " ,including an implementation of imap for MappedGeometry"
            )
        except NotImplementedError as e:
            raise NotImplementedError(error_message)

        # Check for other errors that may prevent computing the gradient
        self._check_gradient_can_be_computed(direction, kwargs)

        # Also obtain the function values representation of the variables
        kwargs_fun = self._2fun(
            geometry=self.domain_geometry, is_par=is_var_par, **kwargs
        )

        # Store if any of the inputs is a CUQIarray
        to_CUQIarray = isinstance(direction, CUQIarray) or any(
            isinstance(x, CUQIarray) for x in kwargs_fun.values()
        )

        # Turn to_CUQIarray to a tuple of bools of the same length as kwargs_fun
        to_CUQIarray = tuple([to_CUQIarray] * len(kwargs_fun))

        # Convert direction to function value
        direction_fun = self._2fun(
            direction=direction, geometry=self.range_geometry, is_par=is_direction_par
        )

        # If model have _original_non_default_args, we use it to replace the
        # kwargs keys so that it matches self._gradient_func signature
        if hasattr(self, "_original_non_default_args"):
            args_fun = list(kwargs_fun.values())
            kwargs_fun = {
                k: v for k, v in zip(self._original_non_default_args, args_fun)
            }
        # Append the direction to the kwargs_fun as first input
        kwargs_fun_grad_input = {**direction_fun, **kwargs_fun}

        # From 1 of gradient (callable)
        if callable(self._gradient_func):
            grad = self._gradient_func(**kwargs_fun_grad_input)
            grad_is_par = False  # Assume gradient is function value

        # Form 2 of gradient (tuple of callables)
        elif isinstance(self._gradient_func, tuple):
            grad = []
            for i, grad_func in enumerate(self._gradient_func):
                if grad_func is not None:
                    grad.append(grad_func(**kwargs_fun_grad_input))
                else:
                    grad.append(None)
                    # set the ith item of to_CUQIarray tuple to False
                    # because the ith gradient is None
                    to_CUQIarray = to_CUQIarray[:i] + (False,) + to_CUQIarray[i + 1 :]
            grad_is_par = False  # Assume gradient is function value

        grad = self._apply_chain_rule_to_account_for_domain_geometry_gradient(
            kwargs_par, grad, grad_is_par, to_CUQIarray
        )

        if len(grad) == 1:
            return list(grad.values())[0]
        return grad

    def _check_gradient_can_be_computed(self, direction, kwargs_dict):
        """Private function that checks if the gradient can be computed. By
        raising an error for the cases where the gradient cannot be computed."""

        # Raise an error if _gradient_func function is not set
        if self._gradient_func is None:
            raise NotImplementedError("Gradient is not implemented for this model.")

        # Raise an error if either the direction or kwargs are Samples objects
        if isinstance(direction, Samples) or any(
            isinstance(x, Samples) for x in kwargs_dict.values()
        ):
            raise NotImplementedError(
                "Gradient is not implemented for input of type Samples."
            )

        # Raise an error if range_geometry is not in the list returned by
        # `_get_identity_geometries()`. i.e. The Jacobian of its
        # par2fun map is not identity.
        # TODO: Add range geometry gradient to the chain rule
        if not type(self.range_geometry) in _get_identity_geometries():
            raise NotImplementedError(
                "Gradient is not implemented for model {} with range geometry {}. You can use one of the geometries in the list {}.".format(
                    self,
                    self.range_geometry,
                    [i_g.__name__ for i_g in _get_identity_geometries()],
                )
            )

        # Raise an error if domain_geometry (or its components in case of
        # _ProductGeometry) does not have gradient attribute and is not in the
        # list returned by `_get_identity_geometries()`. i.e. The Jacobian of its
        # par2fun map is not identity.
        domain_geometries = (
            self.domain_geometry.geometries
            if isinstance(
                self.domain_geometry, cuqi.experimental.geometry._ProductGeometry
            )
            else [self.domain_geometry]
        )
        for domain_geometry in domain_geometries:
            if (
                not hasattr(domain_geometry, "gradient")
                and not type(domain_geometry) in _get_identity_geometries()
            ):
                raise NotImplementedError(
                    "Gradient is not implemented for model \n{}\nwith domain geometry (or domain geometry component) {}. The domain geometries should have gradient method or be from the geometries in the list {}.".format(
                        self,
                        domain_geometry,
                        [i_g.__name__ for i_g in _get_identity_geometries()],
                    )
                )

    def _apply_chain_rule_to_account_for_domain_geometry_gradient(self,
                                                                  kwargs_par,
                                                                  grad,
                                                                  grad_is_par,
                                                                  to_CUQIarray):
        """ Private function that applies the chain rule to account for the
        gradient of the domain geometry. """
        # Create list of domain geometries
        geometries = (
            self.domain_geometry.geometries
            if isinstance(self.domain_geometry, cuqi.experimental.geometry._ProductGeometry)
            else [self.domain_geometry]
        )

        # turn grad_is_par to a tuple of bools if it is not already
        if isinstance(grad_is_par, bool):
            grad_is_par = tuple([grad_is_par]*len(grad))

        # If the domain geometry is a _ProductGeometry and the gradient is
        # stacked, split it
        if (
            isinstance(
                self.domain_geometry, cuqi.experimental.geometry._ProductGeometry
            )
            and not isinstance(grad, (list, tuple))
            and isinstance(grad, np.ndarray)
        ):
            grad = np.split(grad, self.domain_geometry.stacked_par_split_indices)

        # If the domain geometry is not a _ProductGeometry, turn grad into a
        # list of length 1, so that we can iterate over it
        if not isinstance(self.domain_geometry, cuqi.experimental.geometry._ProductGeometry):
            grad = [grad]

        # apply the gradient of each geometry component
        grad_kwargs = {}
        for i, (k, v_par) in enumerate(kwargs_par.items()):
            if hasattr(geometries[i], 'gradient') and grad[i] is not None:
                grad_kwargs[k] = geometries[i].gradient(grad[i], v_par)
                # update the ith component of grad_is_par to True
                grad_is_par = grad_is_par[:i] + (True,) + grad_is_par[i+1:]
            else:
                grad_kwargs[k] = grad[i]

        # convert the computed gradient to parameters
        grad = self._2par(geometry=self.domain_geometry,
                          to_CUQIarray=to_CUQIarray,
                          is_par=grad_is_par,
                          **grad_kwargs)

        return grad

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

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

        # If linear operator is of type Model, it needs to be a LinearModel
        if isinstance(linear_operator, Model) and not isinstance(
            linear_operator, LinearModel
        ):
            raise TypeError(
                "The linear operator should be a LinearModel object, a callable function or a matrix."
            )

        # If the adjoint operator is of type Model, it needs to be a LinearModel
        if isinstance(linear_operator_adjoint, Model) and not isinstance(
            linear_operator_adjoint, LinearModel
        ):
            raise TypeError(
                "The adjoint linear operator should be a LinearModel object, a callable function or a matrix."
            )

        # Additional checks if the linear_operator is not a LinearModel:
        if not isinstance(linear_operator, LinearModel):
            # Ensure the linear operator has exactly one input argument
            if len(cuqi.utilities.get_non_default_args(linear_operator)) != 1:
                raise ValueError(
                    "The linear operator should have exactly one input argument."
                )
            # Ensure the adjoint linear operator has exactly one input argument
            if (
                linear_operator_adjoint is not None
                and len(cuqi.utilities.get_non_default_args(linear_operator_adjoint))
                != 1
            ):
                raise ValueError(
                    "The adjoint linear operator should have exactly one input argument."
                )

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
        self._gradient_func = lambda direction, *args, **kwargs: linear_operator_adjoint(direction)

        # Update forward function to include shift (overwriting the one from Model class)
        self._forward_func = lambda *args, **kwargs: linear_operator(*args, **kwargs) + shift

        # Set stored_non_default_args to None
        self._stored_non_default_args = None

    @property
    def _non_default_args(self):
        if self._stored_non_default_args is None:
            # Use arguments from user's callable linear operator
            self._stored_non_default_args = cuqi.utilities.get_non_default_args(
                self._linear_operator
            )
        return self._stored_non_default_args

    @property
    def shift(self):
        """ The shift of the affine model. """
        return self._shift

    @shift.setter
    def shift(self, value):
        """ Update the shift of the affine model. Updates both the shift value and the underlying forward function. """
        self._shift = value
        self._forward_func = lambda *args, **kwargs: self._linear_operator(*args, **kwargs) + value

    def _forward_func_no_shift(self, *args, is_par=True, **kwargs):
        """Helper function for computing the forward operator without the shift."""
        # convert args to kwargs
        kwargs = self._parse_args_add_to_kwargs(*args, **kwargs, is_par=is_par)
        return self._apply_func(self._linear_operator, **kwargs, is_par=is_par)

    def _adjoint_func_no_shift(self, *args, is_par=True, **kwargs):
        """Helper function for computing the adjoint operator without the shift."""
        # convert args to kwargs
        kwargs = self._parse_args_add_to_kwargs(
            *args,
            **kwargs,
            is_par=is_par,
            non_default_args=cuqi.utilities.get_non_default_args(
                self._linear_operator_adjoint
            ),
        )
        return self._apply_func(
            self._linear_operator_adjoint, **kwargs, is_par=is_par, fwd=False
        )


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

        # Initialize as AffineModel with shift=0
        super().__init__(forward, 0, adjoint, range_geometry, domain_geometry)

    def adjoint(self, *args, is_par=True, **kwargs):
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
        kwargs = self._parse_args_add_to_kwargs(
            *args,
            **kwargs,
            is_par=is_par,
            non_default_args=cuqi.utilities.get_non_default_args(
                self._linear_operator_adjoint
            ),
        )

        # length of kwargs should be 1
        if len(kwargs) > 1:
            raise ValueError(
                "The adjoint operator input is specified by more than one argument. This is not supported."
            )
        if self._linear_operator_adjoint is None:
            raise ValueError("No adjoint operator was provided for this model.")
        return self._apply_func(
            self._linear_operator_adjoint, **kwargs, is_par=is_par, fwd=False
        )

    def __matmul__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def get_matrix(self):
        """
        Returns an ndarray with the matrix representing the forward operator.
        """
        if self._matrix is not None: #Matrix exists so return it
            return self._matrix
        else:
            # TODO: Can we compute this faster while still in sparse format?
            mat = csc_matrix((self.range_dim,0)) #Sparse (m x 1 matrix)
            e = np.zeros(self.domain_dim)

            # Stacks sparse matrices on csc matrix
            for i in range(self.domain_dim):
                e[i] = 1
                col_vec = self.forward(e)
                mat = hstack((mat,col_vec[:,None])) #mat[:,i] = self.forward(e)
                e[i] = 0

            # Store matrix for future use
            self._matrix = mat

            return self._matrix   

    @property
    def T(self):
        """Transpose of linear model. Returns a new linear model acting as the transpose."""
        transpose = LinearModel(
            self._linear_operator_adjoint,
            self._linear_operator,
            self.domain_geometry,
            self.range_geometry,
        )
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
        # PDE needs to be set before calling super().__init__
        # for the property _non_default_args to work
        self.pde = PDE
        self._stored_non_default_args = None

        super().__init__(self._forward_func, range_geometry, domain_geometry)

    @property
    def _non_default_args(self):
        if self._stored_non_default_args is None:    
            # extract the non-default arguments of the PDE
            self._stored_non_default_args = cuqi.utilities.get_non_default_args(self.pde.PDE_form)
            # remove t from the non-default arguments
            self._stored_non_default_args = self._non_default_args
            if 't' in self._non_default_args:
                self._stored_non_default_args.remove('t')

        return self._stored_non_default_args

    def _forward_func(self, **kwargs):
        
        self.pde.assemble(**kwargs)

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
