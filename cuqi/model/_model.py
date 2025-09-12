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
from cuqi.utilities import force_ndarray

class Model(object):
    """Generic model defined by a forward operator.

    Parameters
    -----------
    forward : callable function
        Forward operator of the model. It takes one or more inputs and returns the model output.

    range_geometry : integer, a 1D or 2D tuple of integers, cuqi.geometry.Geometry 
        If integer or 1D tuple of integers is given, a cuqi.geometry._DefaultGeometry1D is created with dimension of the integer.
        If 2D tuple of integers is given, a cuqi.geometry._DefaultGeometry2D is created with dimensions of the tuple.
        If cuqi.geometry.Geometry object is given, it is used as the range geometry of the model.

    domain_geometry : integer, a 1D or 2D tuple of integers, cuqi.geometry.Geometry or a tuple with items of any of the listed types
        If integer or 1D tuple of integers is given, a cuqi.geometry._DefaultGeometry1D is created with dimension of the integer.
        If 2D tuple of integers is given (and the forward model has one input only), a cuqi.geometry._DefaultGeometry2D is created with dimensions of the tuple.
        If cuqi.geometry.Geometry is given, it is used as the domain geometry.
        If tuple of the above types is given, a cuqi.geometry._ProductGeometry is created based on the tuple entries. This is used for models with multiple inputs where each entry in the tuple represents the geometry of each input.

    gradient : callable function, a tuple of callable functions or None, optional
        The direction-Jacobian product of the forward model Jacobian with respect to the model input, evaluated at the model input. For example, if the forward model inputs are `x` and `y`, the gradient callable signature should be (`direction`, `x`, `y`), in that order, where `direction` is the direction by which the Jacobian matrix is multiplied and `x` and `y` are the parameters at which the Jacobian is computed.

        If the gradient function is a single callable function, it returns a 1D ndarray if the model has only one input. If the model has multiple inputs, this gradient function should return a tuple of 1D ndarrays, each representing the gradient with respect to each input.

        If the gradient function is a tuple of callable functions, each callable function should return a 1D ndarray representing the gradient with respect to each input. The order of the callable functions in the tuple should match the order of the model inputs.

    jacobian : callable function, a tuple of callable functions or None, optional
        The Jacobian of the forward model with respect to the forward model input, evaluated at the model input. For example, if the forward model inputs are `x` and `y`, the jacobian signature should be (`x`, `y`), in that order, where `x` and `y` are the parameters at which the Jacobian is computed.

        If the Jacobian function is a single callable function, it should return a 2D ndarray of shape (range_dim, domain_dim) if the model has only one input. If the model has multiple inputs, this Jacobian function should return a tuple of 2D ndarrays, each representing the Jacobian with respect to each input. 
        
        If the Jacobian function is a tuple of callable functions, each callable function should return a 2D ndarray representing the Jacobian with respect to each input. The order of the callable functions in the tuple should match the order of the model inputs.
    
        The Jacobian function is used to specify the gradient function by computing the vector-Jacobian product (VJP), here we refer to the vector in the VJP as the `direction` since it is the direction at which the gradient is computed. Either the gradient or the Jacobian can be specified, but not both.


    :ivar range_geometry: The geometry representing the range.
    :ivar domain_geometry: The geometry representing the domain.

    Example 1
    ----------

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

        print(model(np.array([1, 1])))
        print(model.gradient(np.array([1]), np.array([1, 1])))

    Alternatively, the gradient information in the forward model can be defined by direction-Jacobian product using the gradient keyword argument.

    This may be more efficient if forming the Jacobian matrix is expensive.

    .. code-block:: python

        import numpy as np
        from cuqi.model import Model

        def forward(x):
            return 10*x[1] - 10*x[0]**3 + 5*x[0]**2 + 6*x[0]

        def gradient(direction, x):
            # Direction-Jacobian product direction@jacobian(x)
            return direction@np.array([[-30*x[0]**2 + 10*x[0] + 6, 10]])

        model = Model(forward, range_geometry=1, domain_geometry=2, gradient=gradient)

        print(model(np.array([1, 1])))
        print(model.gradient(np.array([1]), np.array([1, 1])))

    Example 2
    ----------
    Alternatively, the example above can be defined as a model with multiple inputs: :math:`x` and :math:`y`:

    .. code-block:: python

        import numpy as np
        from cuqi.model import Model
        from cuqi.geometry import Discrete
    
        def forward(x, y):
            return 10 * y - 10 * x**3 + 5 * x**2 + 6 * x
    
        def jacobian(x, y):
            return (np.array([[-30 * x**2 + 10 * x + 6]]), np.array([[10]]))
    
        model = Model(
            forward,
            range_geometry=1,
            domain_geometry=(Discrete(1), Discrete(1)),
            jacobian=jacobian,
        )
    
        print(model(1, 1))
        print(model.gradient(np.array([1]), 1, 1))
    """

    _supports_partial_eval = True
    """Flag indicating that partial evaluation of Model objects is supported, i.e., calling the model object with only some of the inputs specified returns a model that can be called with the remaining inputs."""

    def __init__(self, forward, range_geometry, domain_geometry, gradient=None, jacobian=None):

        # Check if input is callable
        if callable(forward) is not True:
            raise TypeError("Forward needs to be callable function.")

        # Store forward func
        self._forward_func = forward
        self._stored_non_default_args = None

        # Store range_geometry
        self.range_geometry = range_geometry

        # Store domain_geometry
        self.domain_geometry = domain_geometry

        # Additional checks for the forward operator
        self._check_domain_geometry_consistent_with_forward()

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

        # Set gradient output stacked flag to False
        self._gradient_output_stacked = False

    @property
    def _non_default_args(self):
        if self._stored_non_default_args is None:
            # Store non_default_args of the forward operator for faster caching
            # when checking for those arguments.
            self._stored_non_default_args =\
                cuqi.utilities.get_non_default_args(self._forward_func)
        return self._stored_non_default_args

    @property
    def number_of_inputs(self):
        """ The number of inputs of the model. """
        return len(self._non_default_args)

    @property
    def range_geometry(self):
        """ The geometry representing the range of the model. """
        return self._range_geometry

    @range_geometry.setter
    def range_geometry(self, value):
        """ Update the range geometry of the model. """
        if isinstance(value, Geometry):
            self._range_geometry = value
        elif isinstance(value, int):
            self._range_geometry = self._create_default_geometry(value)
        elif isinstance(value, tuple):
            self._range_geometry = self._create_default_geometry(value)
        elif value is None:
            raise AttributeError(
                "The parameter 'range_geometry' is not specified by the user and it cannot be inferred from the attribute 'forward'."
            )
        else:
            raise TypeError(
                " The allowed types for 'range_geometry' are: 'cuqi.geometry.Geometry', int, 1D tuple of int, or 2D tuple of int."
            )

    @property
    def domain_geometry(self):
        """ The geometry representing the domain of the model. """
        return self._domain_geometry

    @domain_geometry.setter
    def domain_geometry(self, value):
        """ Update the domain geometry of the model. """

        if isinstance(value, Geometry):
            self._domain_geometry = value
        elif isinstance(value, int):
            self._domain_geometry = self._create_default_geometry(value)
        elif isinstance(value, tuple) and self.number_of_inputs == 1:
            self._domain_geometry = self._create_default_geometry(value)
        elif isinstance(value, tuple) and self.number_of_inputs > 1:
            geometries = [item if isinstance(item, Geometry) else self._create_default_geometry(item) for item in value]
            self._domain_geometry = cuqi.experimental.geometry._ProductGeometry(*geometries)
        elif value is None:
            raise AttributeError(
                "The parameter 'domain_geometry' is not specified by the user and it cannot be inferred from the attribute 'forward'."
            )
        else:
            raise TypeError(
                "For forward model with 1 input, the allowed types for 'domain_geometry' are: 'cuqi.geometry.Geometry', int, 1D tuple of int, or 2D tuple of int. For forward model with multiple inputs, the 'domain_geometry' should be a tuple with items of any of the above types."
            )

    def _create_default_geometry(self, value):
        """Private function that creates default geometries for the model."""
        if isinstance(value, tuple) and len(value) == 1:
            value = value[0]
        if isinstance(value, Geometry):
            return value
        if isinstance(value, int):
            return _DefaultGeometry1D(grid=value)
        elif isinstance(value, tuple) and len(value) == 2:
            return _DefaultGeometry2D(im_shape=value)
        else:
            raise ValueError(
                "Default geometry creation can be specified by an integer or a 2D tuple of integers."
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

    def _check_domain_geometry_consistent_with_forward(self):
        """Private function that checks if the domain geometry of the model is
        consistent with the forward operator."""
        if (
            not isinstance(
                self.domain_geometry, cuqi.experimental.geometry._ProductGeometry
            )
            and self.number_of_inputs > 1
        ):
            raise ValueError(
                "The forward operator input is specified by more than one argument. This is only supported for domain geometry of type tuple with items of type: cuqi.geometry.Geometry object, int, or 2D tuple of int."
            )

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
            if len(func) != self.number_of_inputs:
                raise ValueError(
                    f"The "
                    + func_type.lower()
                    + f" tuple length should be {self.number_of_inputs} for model with inputs {self._non_default_args}"
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

        expected_func_non_default_args = (
            self._non_default_args
            if not hasattr(self, "_original_non_default_args")
            else self._original_non_default_args
        )

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

                if list(func_non_default_args) != list(expected_func_non_default_args):
                    raise ValueError(
                        func_type.capitalize()
                        + f" function signature should be {expected_func_non_default_args}"
                    )

    def _use_jacobian_to_specify_gradient(self, jacobian):
        """Private function that uses the jacobian function to specify the
        gradient function."""
        # if jacobian is a single function and model has multiple inputs
        if callable(jacobian) and self.number_of_inputs > 1:
            gradient = self._create_gradient_lambda_function_from_jacobian_with_correct_signature(
                jacobian, form='one_callable_multiple_inputs'
            )
        # Elif jacobian is a single function and model has only one input
        elif callable(jacobian):
            gradient = self._create_gradient_lambda_function_from_jacobian_with_correct_signature(
                jacobian, form='one_callable_one_input'
            )
        # Else, jacobian is a tuple of jacobian functions
        else:
            gradient = []
            for jac in jacobian:
                if jac is not None:
                    gradient.append(
                        self._create_gradient_lambda_function_from_jacobian_with_correct_signature(
                            jac, form='tuple_of_callables'
                        )
                    )
                else:
                    gradient.append(None)
        return tuple(gradient) if isinstance(gradient, list) else gradient

    def _create_gradient_lambda_function_from_jacobian_with_correct_signature(
        self, jacobian, form
    ):
        """Private function that creates gradient lambda function from the
        jacobian function, with the correct signature (based on the model
        non_default_args).
        """
        # create the string representation of the lambda function
        # for different forms of jacobian
        if form=='one_callable_multiple_inputs':
            grad_fun_str = (
                "lambda direction, "
                + ", ".join(self._non_default_args)
                + ", jacobian: tuple([direction@jacobian("
                + ", ".join(self._non_default_args)
                + ")[i] for i in range("+str(self.number_of_inputs)+")])"
            )
        elif form=='tuple_of_callables' or form=='one_callable_one_input':
            grad_fun_str = (
                "lambda direction, "
                + ", ".join(self._non_default_args)
                + ", jacobian: direction@jacobian("
                + ", ".join(self._non_default_args)
                + ")"
            )
        else:
            raise ValueError("form should be either 'one_callable' or 'tuple_of_callables'.")

        # create the lambda function from the string
        grad_func = eval(grad_fun_str)

        # create partial function from the lambda function with jacobian as a
        # fixed argument
        grad_func = partial(grad_func, jacobian=jacobian)

        return grad_func

    def _2fun(self, geometry=None, is_par=True, **kwargs):
        """ Converts `kwargs` to function values (if needed) using the geometry. For example, `kwargs` can be the model input which need to be converted to function value before being passed to :class:`~cuqi.model.Model` operators (e.g. _forward_func, _adjoint_func, _gradient_func).

        Parameters
        ----------
        geometry : cuqi.geometry.Geometry
            The geometry representing the values in `kwargs`.

        is_par : bool or a tuple of bools
            If `is_par` is True, the values in `kwargs` are assumed to be parameters.
            If `is_par` is False, the values in `kwargs` are assumed to be function values.
            If `is_par` is a tuple of bools, the values in `kwargs` are assumed to be parameters or function values based on the corresponding boolean value in the tuple.
        
        **kwargs : keyword arguments to be converted to function values.

        Returns
        -------
        dict of the converted values
        """
        # Check kwargs and geometry are consistent and set up geometries list and
        # is_par tuple
        geometries, is_par = self._helper_pre_conversion_checks_and_processing(geometry, is_par, **kwargs)

        # Convert to function values
        for i, (k, v) in enumerate(kwargs.items()):
            # Use CUQIarray funvals if geometry is consistent
            if isinstance(v, CUQIarray) and v.geometry == geometries[i]:
                kwargs[k] = v.funvals
            # Else, if we still need to convert to function value (is_par[i] is True)
            # we use the geometry par2fun method
            elif is_par[i] and v is not None:
                kwargs[k] = geometries[i].par2fun(v)
            else:
                # No need to convert
                pass

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
        """ Converts `kwargs` to parameters using the geometry. For example, `kwargs` can be the output of :class:`~cuqi.model.Model` operators (e.g. _forward_func, _adjoint_func, _gradient_func) which need to be converted to parameters before being returned.

        Parameters
        ----------
        geometry : cuqi.geometry.Geometry
            The geometry representing the values in `kwargs`.

        to_CUQIarray : bool or a tuple of bools
            If `to_CUQIarray` is True, the values in `kwargs` will be wrapped in `CUQIarray`.
            If `to_CUQIarray` is False, the values in `kwargs` will not be wrapped in `CUQIarray`.
            If `to_CUQIarray` is a tuple of bools, the values in `kwargs` will be wrapped in `CUQIarray` or not based on the corresponding boolean value in the tuple.
        
        is_par : bool or a tuple of bools
            If `is_par` is True, the values in `kwargs` are assumed to be parameters.
            If `is_par` is False, the values in `kwargs` are assumed to be function values.
            If `is_par` is a tuple of bools, the values in `kwargs` are assumed to be parameters or function values based on the corresponding boolean value in the tuple.

        Returns
        -------
        dict of the converted values
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
            # Else, if we still need to convert to parameter value (is_par[i] is False)
            # we use the geometry fun2par method
            elif not is_par[i] and v is not None:
                v = geometries[i].fun2par(v)
            else:
                # No need to convert
                pass

            # Wrap the value v in CUQIarray if requested
            if to_CUQIarray[i] and v is not None:
                v = CUQIarray(v, is_par=True, geometry=geometries[i])

            kwargs[k] = v

        return kwargs

    def _apply_func(self, func=None, fwd=True, is_par=True, **kwargs):
        """ Private function that applies the given function `func` to the input `kwargs`. It converts the input to function values (if needed) and converts the output to parameter values. It additionally handles the case of applying the function `func` to cuqi.samples.Samples objects.
        
        Parameters
        ----------
        func: function handler 
            The function to be applied.

        fwd : bool
            Flag indicating the direction of the operator to determine the range and domain geometries of the function.
            If True the function is a forward operator.
            If False the function is an adjoint operator.

        is_par : bool or list of bool
            If True, the inputs in `kwargs` are assumed to be parameters.
            If False, the input in `kwargs` are assumed to be function values.
            If `is_par` is a list of bools, the inputs are assumed to be parameters or function values based on the corresponding boolean value in the list.

        Returns
        -------
        ndarray or cuqi.array.CUQIarray or cuqi.samples.Samples object
            The output of the function.
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
            return self._handle_case_when_model_input_is_samples(func, fwd, **kwargs)

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

    def _handle_case_when_model_input_is_samples(self, func=None, fwd=True, **kwargs):
        """Private function that calls apply_func for samples in the
        Samples object(s).
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
            kwargs_i = {
                k: CUQIarray(v.samples[..., i], is_par=v.is_par, geometry=v.geometry)
                for k, v in kwargs.items()
            }
            out[:, i] = self._apply_func(func=func, fwd=fwd, **kwargs_i)
        # Specify the range geometries of the function
        func_range_geometry = self.range_geometry if fwd else self.domain_geometry
        return Samples(out, geometry=func_range_geometry)

    def _parse_args_add_to_kwargs(
        self, *args, is_par=True, non_default_args=None, map_name="model", **kwargs
    ):
        """ Private function that parses the input arguments and adds them as
        keyword arguments matching (the order of) the non default arguments of
        the forward function or other specified non_default_args list.
        """
        # If non_default_args is not specified, use the non_default_args of the
        # model
        if non_default_args is None:
            non_default_args = self._non_default_args

        # Either args or kwargs can be provided but not both
        if len(args) > 0 and len(kwargs) > 0:
            raise ValueError(
                "The "
                + map_name.lower()
                + " input is specified both as positional and keyword arguments. This is not supported."
            )

        len_input = len(args) + len(kwargs)

        # If partial evaluation, make sure input is not of type Samples
        if len_input < len(non_default_args):
            # If the argument is a Sample object, splitting or partial
            # evaluation of the model is not supported
            temp_args = args if len(args) > 0 else list(kwargs.values())
            if any(isinstance(arg, Samples) for arg in temp_args):
                raise ValueError(("When using Samples objects as input, the"
                                +" user should provide a Samples object for"
                                +f" each non_default_args {non_default_args}"
                                +" of the model. That is, partial evaluation"
                                +" or splitting is not supported for input"
                                +" of type Samples."))

        # If args are given, add them to kwargs
        if len(args) > 0:

            # Check if the input is for multiple input case and is stacked,
            # then split it
            if len(args) < len(non_default_args):
                args = self._split_in_case_of_stacked_args(*args, is_par=is_par)

            # Add args to kwargs following the order of non_default_args
            for idx, arg in enumerate(args):
                kwargs[non_default_args[idx]] = arg
    
        # Check kwargs matches non_default_args
        if not (set(list(kwargs.keys())) <= set(non_default_args)):
            if map_name == "gradient":
                error_msg = f"The gradient input is specified by a direction and keywords arguments {list(kwargs.keys())} that does not match the non_default_args of the model {non_default_args}."
            else:
                error_msg = (
                    "The "
                    + map_name.lower()
                    + f" input is specified by keywords arguments {list(kwargs.keys())} that does not match the non_default_args of the "
                    + map_name
                    + f" {non_default_args}."
                )

            raise ValueError(error_msg)

        # Make sure order of kwargs is the same as non_default_args
        kwargs = {k: kwargs[k] for k in non_default_args if k in kwargs}

        return kwargs

    def _split_in_case_of_stacked_args(self, *args, is_par=True):
        """Private function that checks if the input args is a stacked
        CUQIarray or numpy array and splits it into multiple arguments based on
        the domain geometry of the model. Otherwise, it returns the input args
        unchanged."""

        # Check conditions for splitting and split if all conditions are met
        is_CUQIarray = isinstance(args[0], CUQIarray)
        is_numpy_array = isinstance(args[0], np.ndarray)

        if ((is_CUQIarray or is_numpy_array) and
           is_par and
           len(args) == 1 and
           args[0].shape == (self.domain_dim,) and
           isinstance(self.domain_geometry, cuqi.experimental.geometry._ProductGeometry)):
            # Split the stacked input
            split_args = np.split(args[0], self.domain_geometry.stacked_par_split_indices)
            # Convert split args to CUQIarray if input is CUQIarray
            if is_CUQIarray:
                split_args = [
                    CUQIarray(arg, is_par=True, geometry=self.domain_geometry.geometries[i])
                    for i, arg in enumerate(split_args)
                ]
            return split_args

        else:
            return args

    def forward(self, *args, is_par=True, **kwargs):
        """ Forward function of the model.

        Forward converts the input to function values (if needed) using the domain geometry of the model. Then it applies the forward operator to the function values and converts the output to parameters using the range geometry of the model.

        Parameters
        ----------
        *args : ndarrays or cuqi.array.CUQIarray objects or cuqi.samples.Samples objects
            Positional arguments for the forward operator. The forward operator input can be specified as either positional arguments or keyword arguments but not both.

            If the input is specified as positional arguments, the order of the arguments should match the non_default_args of the model.

        is_par : bool or a tuple of bools
            If True, the inputs in `args` or `kwargs` are assumed to be parameters.
            If False, the inputs in `args` or `kwargs` are assumed to be function values.
            If `is_par` is a tuple of bools, the inputs are assumed to be parameters or function values based on the corresponding boolean value in the tuple.

        **kwargs : keyword arguments
            keyword arguments for the forward operator. The forward operator input can be specified as either positional arguments or keyword arguments but not both.

            If the input is specified as keyword arguments, the keys should match the non_default_args of the model.

        Returns
        -------
        ndarray or cuqi.array.CUQIarray or cuqi.samples.Samples object
            The model output. Always returned as parameters.
        """

        # Add args to kwargs and ensure the order of the arguments matches the
        # non_default_args of the forward function
        kwargs = self._parse_args_add_to_kwargs(
            *args, **kwargs, is_par=is_par, map_name="model"
        )
        # Extract args from kwargs
        args = list(kwargs.values())

        if len(kwargs) == 0:
            return self

        partial_arguments = len(kwargs) < len(self._non_default_args)

        # If input is a distribution, we simply change the parameter name of
        # model to match the distribution name
        if all(isinstance(x, cuqi.distribution.Distribution)
               for x in kwargs.values()):
            if partial_arguments:
                raise ValueError(
                    "Partial evaluation of the model is not supported for distributions."
                )
            return self._handle_case_when_model_input_is_distributions(kwargs)

        # If input is a random variable, we handle it separately
        elif all(isinstance(x, cuqi.experimental.algebra.RandomVariable)
               for x in kwargs.values()):
            if partial_arguments:
                raise ValueError(
                    "Partial evaluation of the model is not supported for random variables."
                )
            return self._handle_case_when_model_input_is_random_variables(kwargs)

        # If input is a Node from internal abstract syntax tree, we let the Node handle the operation
        # We use NotImplemented to indicate that the operation is not supported from the Model class
        # in case of operations such as "@" that can be interpreted as both __matmul__ and __rmatmul__
        # the operation may be delegated to the Node class.
        elif any(isinstance(args_i, cuqi.experimental.algebra.Node) for args_i in args):
            return NotImplemented

        # if input is partial, we create a new model with the partial input
        if partial_arguments:
            # Create is_par_partial from the is_par to contain only the relevant parts
            if isinstance(is_par, (list, tuple)):
                is_par_partial = tuple(
                    is_par[i]
                    for i in range(self.number_of_inputs)
                    if self._non_default_args[i] in kwargs.keys()
                )
            else:
                is_par_partial = is_par
            # Build a partial model with the given kwargs
            partial_model = self._build_partial_model(kwargs, is_par_partial)
            return partial_model

        # Else we apply the forward operator
        # if model has _original_non_default_args, we use it to replace the
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

    def _build_partial_model(self, kwargs, is_par):
        """Private function that builds a partial model substituting the given
        keyword arguments with their values. The created partial model will have
        as inputs the non-default arguments that are not in the kwargs."""

        # Extract args from kwargs
        args = list(kwargs.values())

        # Define original_non_default_args which represents the complete list of
        # non-default arguments of the forward function.
        original_non_default_args = (
            self._original_non_default_args
            if hasattr(self, "_original_non_default_args")
            else self._non_default_args
        )

        if hasattr(self, "_original_non_default_args"):
            # Split the _original_non_default_args into two lists:
            # 1. reduced_original_non_default_args: the _original_non_default_args
            # corresponding to the _non_default_args that are not in kwargs
            # 2. substituted_non_default_args: the _original_non_default_args
            # corresponding to the _non_default_args that are in kwargs
            reduced_original_non_default_args = [
                original_non_default_args[i]
                for i in range(self.number_of_inputs)
                if self._non_default_args[i] not in kwargs.keys()
            ]
            substituted_non_default_args = [
                original_non_default_args[i]
                for i in range(self.number_of_inputs)
                if self._non_default_args[i] in kwargs.keys()
            ]
            # Replace the keys in kwargs with the substituted_non_default_args
            # so that the kwargs match the signature of the _forward_func
            kwargs = {k: v for k, v in zip(substituted_non_default_args, args)}

        # Create a partial domain geometry with the geometries corresponding
        # to the non-default arguments that are not in kwargs (remaining
        # unspecified inputs)
        partial_domain_geometry = cuqi.experimental.geometry._ProductGeometry(
            *[
                self.domain_geometry.geometries[i]
                for i in range(self.number_of_inputs)
                if original_non_default_args[i] not in kwargs.keys()
            ]
        )

        if len(partial_domain_geometry.geometries) == 1:
            partial_domain_geometry = partial_domain_geometry.geometries[0]

        # Create a domain geometry with the geometries corresponding to the
        # non-default arguments that are specified
        substituted_domain_geometry = cuqi.experimental.geometry._ProductGeometry(
            *[
                self.domain_geometry.geometries[i]
                for i in range(self.number_of_inputs)
                if original_non_default_args[i] in kwargs.keys()
            ]
        )

        if len(substituted_domain_geometry.geometries) == 1:
            substituted_domain_geometry = substituted_domain_geometry.geometries[0]

        # Create new model with partial input
        # First, we convert the input to function values
        kwargs = self._2fun(geometry=substituted_domain_geometry, is_par=is_par, **kwargs)

        # Second, we create a partial function for the forward operator
        partial_forward = partial(self._forward_func, **kwargs)

        # Third, if applicable, we create a partial function for the gradient
        if isinstance(self._gradient_func, tuple):
            # If gradient is a tuple, we create a partial function for each
            # gradient function in the tuple
            partial_gradient = tuple(
                (
                    partial(self._gradient_func[i], **kwargs)
                    if self._gradient_func[i] is not None
                    else None
                )
                for i in range(self.number_of_inputs)
                if original_non_default_args[i] not in kwargs.keys()
            )
            if len(partial_gradient) == 1:
                partial_gradient = partial_gradient[0]

        elif callable(self._gradient_func):
            raise NotImplementedError(
                "Partial forward model is only supported for gradient/jacobian functions that are tuples of callable functions."
            )

        else:
            partial_gradient = None

        # Lastly, we create the partial model with the partial forward
        # operator (we set the gradient function later)
        partial_model = Model(
            forward=partial_forward,
            range_geometry=self.range_geometry,
            domain_geometry=partial_domain_geometry,
        )

        # Set the _original_non_default_args (if applicable) and
        # _stored_non_default_args of the partial model
        if hasattr(self, "_original_non_default_args"):
            partial_model._original_non_default_args = reduced_original_non_default_args
        partial_model._stored_non_default_args = [
            self._non_default_args[i]
            for i in range(self.number_of_inputs)
            if original_non_default_args[i] not in kwargs.keys()
        ]

        # Set the gradient function of the partial model
        partial_model._check_correct_gradient_jacobian_form(
            partial_gradient, "gradient"
        )
        partial_model._gradient_func = partial_gradient

        return partial_model

    def _handle_case_when_model_input_is_distributions(self, kwargs):
        """Private function that handles the case of the input being a
        distribution or multiple distributions."""

        if not self._correct_distribution_dimension(kwargs.values()):
            raise ValueError(
                "Attempting to match parameter name of Model with given distribution(s), but distribution(s) dimension(s) does not match model input dimension(s)."
            )
        new_model = copy(self)

        # Store the original non_default_args of the model
        new_model._original_non_default_args = (
            self._original_non_default_args
            if hasattr(self, "_original_non_default_args")
            else self._non_default_args
        )

        # Update the non_default_args of the model to match the distribution
        # names. Defaults to x in the case of only one distribution that has no
        # name
        new_model._stored_non_default_args = [x.name for x in kwargs.values()]

        # If there is a repeated name, raise an error
        if len(set(new_model._stored_non_default_args)) != len(
            new_model._stored_non_default_args
        ):
            raise ValueError(
                "Attempting to match parameter name of Model with given distributions, but distribution names are not unique. Please provide unique names for the distributions."
            )

        return new_model

    def _handle_case_when_model_input_is_random_variables(self, kwargs):
        """ Private function that handles the case of the input being a random variable. """
        # If random variable is not a leaf-type node (e.g. internal node) we return NotImplemented
        if any(not isinstance(x.tree, cuqi.experimental.algebra.VariableNode) for x in kwargs.values()):
            return NotImplemented        

        # Extract the random variable distributions and check dimensions consistency with domain geometry
        distributions = [value.distribution for value in kwargs.values()]
        if not self._correct_distribution_dimension(distributions):
            raise ValueError("Attempting to match parameter name of Model with given random variable(s), but random variable dimension(s) does not match model input dimension(s).")

        new_model = copy(self)

        # Store the original non_default_args of the model
        new_model._original_non_default_args = self._non_default_args

        # Update the non_default_args of the model to match the random variable
        # names. Defaults to x in the case of only one random variable that has
        # no name
        new_model._stored_non_default_args = [x.name for x in distributions]

        # If there is a repeated name, raise an error
        if len(set(new_model._stored_non_default_args)) != len(
            new_model._stored_non_default_args
        ):
            raise ValueError(
                "Attempting to match parameter name of Model with given random variables, but random variables names are not unique. Please provide unique names for the random variables."
            )

        return new_model

    def gradient(
        self, direction, *args, is_direction_par=True, is_var_par=True, **kwargs
    ):
        """Gradient of the forward operator (Direction-Jacobian product)

        The gradient computes the Vector-Jacobian product (VJP) of the forward operator evaluated at the given model input and the given vector (direction).

        Parameters
        ----------
        direction : ndarray or cuqi.array.CUQIarray
            The direction at which to compute the gradient.

        *args : ndarrays or cuqi.array.CUQIarray objects
            Positional arguments for the values at which to compute the gradient. The gradient operator input can be specified as either positional arguments or keyword arguments but not both.

            If the input is specified as positional arguments, the order of the arguments should match the non_default_args of the model.

        is_direction_par : bool
            If True, `direction` is assumed to be parameters.
            If False, `direction` is assumed to be function values.

        is_var_par : bool or a tuple of bools
            If True, the inputs in `args` or `kwargs` are assumed to be parameters.
            If False, the inputs in `args` or `kwargs` are assumed to be function values.
            If `is_var_par` is a tuple of bools, the inputs in `args` or `kwargs` are assumed to be parameters or function values based on the corresponding boolean value in the tuple.
        """
        # Add args to kwargs and ensure the order of the arguments matches the
        # non_default_args of the forward function
        kwargs = self._parse_args_add_to_kwargs(
            *args, **kwargs, is_par=is_var_par, map_name="gradient"
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

        # If model has _original_non_default_args, we use it to replace the
        # kwargs keys so that it matches self._gradient_func signature
        if hasattr(self, '_original_non_default_args'):
            args_fun = list(kwargs_fun.values())
            kwargs_fun = {
                k: v for k, v in zip(self._original_non_default_args, args_fun)
            }
        # Append the direction to the kwargs_fun as first input
        kwargs_fun_grad_input = {**direction_fun, **kwargs_fun}

        # Form 1 of gradient (callable)
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
        elif self._gradient_output_stacked:
            return np.hstack(
                [
                    (
                        v.to_numpy()
                        if isinstance(v, CUQIarray)
                        else force_ndarray(v, flatten=True)
                    )
                    for v in list(grad.values())
                ]
            )

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
        gradient of the domain geometry. That is, it computes the gradient of
        the function values with respect to the parameters values."""
        # Create list of domain geometries
        geometries = (
            self.domain_geometry.geometries
            if isinstance(self.domain_geometry, cuqi.experimental.geometry._ProductGeometry)
            else [self.domain_geometry]
        )

        # turn grad_is_par to a tuple of bools if it is not already
        if isinstance(grad_is_par, bool):
            grad_is_par = tuple([grad_is_par]*self.number_of_inputs)

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
        kwargs = {}
        if self.number_of_inputs > 1:
            pad = " " * len("CUQI {}: ".format(self.__class__.__name__))
            kwargs["pad"]=pad  
        return "CUQI {}: {} -> {}.\n    Forward parameters: {}.".format(self.__class__.__name__,self.domain_geometry.__repr__(**kwargs),self.range_geometry,self._non_default_args)

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

        # Store linear operator privately
        # Note: we need to set the _linear_operator before calling the
        # super().__init__() because it is needed when calling the property
        # _non_default_args within the super().__init__()
        self._linear_operator = linear_operator

        # Initialize Model class
        super().__init__(linear_operator, range_geometry, domain_geometry)

        # Store matrix privately
        self._matrix = matrix

        # Store shift as private attribute
        self._shift = shift


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
        kwargs = self._parse_args_add_to_kwargs(
            *args, **kwargs, map_name="model", is_par=is_par
        )
        args = list(kwargs.values())
        # if model has _original_non_default_args, we use it to replace the
        # kwargs keys so that it matches self._linear_operator signature
        if hasattr(self, '_original_non_default_args'):
            kwargs = {k:v for k,v in zip(self._original_non_default_args, args)}
        return self._apply_func(self._linear_operator, **kwargs, is_par=is_par)

    def _adjoint_func_no_shift(self, *args, is_par=True, **kwargs):
        """Helper function for computing the adjoint operator without the shift."""
        # convert args to kwargs
        kwargs = self._parse_args_add_to_kwargs(
            *args,
            **kwargs,
            map_name='adjoint',
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
        
        Adjoint converts the input to function values (if needed) using the range geometry of the model then applies the adjoint operator to the function values and converts the output function values to parameters using the domain geometry of the model.

        Parameters
        ----------
        *args : ndarrays or cuqi.array.CUQIarray object
            Positional arguments for the adjoint operator ( maximum one argument). The adjoint operator input can be specified as either positional arguments or keyword arguments but not both.

        **kwargs : keyword arguments
            keyword arguments for the adjoint operator (maximum one argument). The adjoint operator input can be specified as either positional arguments or keyword arguments but not both.

            If the input is specified as keyword arguments, the keys should match the non_default_args of the model.

        Returns
        -------
        ndarray or cuqi.array.CUQIarray
            The adjoint model output. Always returned as parameters.
        """
        kwargs = self._parse_args_add_to_kwargs(
            *args,
            **kwargs,
            map_name='adjoint',
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
    In the forward method the PDE is assembled, solved and observed.
    
    Parameters
    -----------
    PDE : cuqi.pde.PDE
        The PDE that specifies the forward operator.

    range_geometry : integer or cuqi.geometry.Geometry, optional
        If integer is given, a cuqi.geometry._DefaultGeometry is created with dimension of the integer.

    domain_geometry : integer or cuqi.geometry.Geometry, optional
        If integer is given, a cuqi.geometry._DefaultGeometry is created with dimension of the integer.


    :ivar range_geometry: The geometry representing the range.
    :ivar domain_geometry: The geometry representing the domain.
    """
    def __init__(self, PDE: cuqi.pde.PDE, range_geometry, domain_geometry, **kwargs):

        if not isinstance(PDE, cuqi.pde.PDE):
            raise ValueError("PDE needs to be a cuqi PDE.")
        # PDE needs to be set before calling super().__init__
        # for the property _non_default_args to work
        self.pde = PDE
        self._stored_non_default_args = None

        # If gradient or jacobian is not provided, we create it from the PDE
        if not np.any([k in kwargs.keys() for k in ["gradient", "jacobian"]]):
            # Create gradient or jacobian function to pass to the Model based on
            # the PDE object. The dictionary derivative_kwarg contains the
            # created function along with the function type (either "gradient"
            # or "jacobian")
            derivative_kwarg = self._create_derivative_function()
            # append derivative_kwarg to kwargs
            kwargs.update(derivative_kwarg)

        super().__init__(forward=self._forward_func_pde,
                         range_geometry=range_geometry,
                         domain_geometry=domain_geometry,
                         **kwargs)

    @property
    def _non_default_args(self):
        if self._stored_non_default_args is None:
            # extract the non-default arguments of the PDE
            self._stored_non_default_args = self.pde._non_default_args

        return self._stored_non_default_args

    def _forward_func_pde(self, **kwargs):

        self.pde.assemble(**kwargs)

        sol, info = self.pde.solve()

        obs = self.pde.observe(sol)

        return obs

    def _create_derivative_function(self):
        """Private function that creates the derivative function (gradient or
        jacobian) based on the PDE object. The derivative function is created as
        a lambda function that takes the direction and the parameters as input 
        and returns the gradient or jacobian of the PDE. This private function
        returns a dictionary with the created function and the function type
        (either "gradient" or "jacobian")."""

        if hasattr(self.pde, "gradient_wrt_parameter"):
            # Build the string that will be used to create the lambda function
            function_str = (
                "lambda direction, "
                + ", ".join(self._non_default_args)
                + ", pde_func: pde_func(direction, "
                + ", ".join(self._non_default_args)
                + ")"
            )

            # create the lambda function from the string
            function = eval(function_str)

            # create partial function from the lambda function with gradient_wrt_parameter
            # as the first argument
            grad_func = partial(function, pde_func=self.pde.gradient_wrt_parameter)

            # Return the gradient function
            return {"gradient": grad_func}

        elif hasattr(self.pde, "jacobian_wrt_parameter"):
            # Build the string that will be used to create the lambda function
            function_str = (
                "lambda "
                + ", ".join(self._non_default_args)
                + ", pde_func: pde_func( "
                + ", ".join(self._non_default_args)
                + ")"
            )

            # create the lambda function from the string
            function = eval(function_str)

            # create partial function from the lambda function with jacobian_wrt_parameter
            # as the first argument
            jacobian_func = partial(function, pde_func=self.pde.jacobian_wrt_parameter)

            # Return the jacobian function
            return {"jacobian": jacobian_func}
        else:
            return {} # empty dictionary if no gradient or jacobian is found

    # Add the underlying PDE class name to the repr.
    def __repr__(self) -> str:
        return super().__repr__()+"\n    PDE: {}.".format(self.pde.__class__.__name__)
