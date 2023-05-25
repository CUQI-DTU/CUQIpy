from __future__ import annotations
from typing import List, Any


class RandomVariable:
    """ Random variable defined by a base distribution with potential algebraic operations applied to it.

    The random variable class is still experimental.

    The random variable can be viewed as a lazily evaluated array. This allows for the definition of
    Bayesian models in a natural way.

    Example
    -------

    Basic usage:

    .. code-block:: python

        from cuqi.distribution import Gaussian

        x = Gaussian(0, 1)

        print(x+1)  # returns a RandomVariable object with operation recorded

    Directly creating random variable:
   
    .. code-block:: python

        from cuqi.distribution import Gaussian

        x = Gaussian(0, 1)

        x_rv = x.as_random_variable()  # returns a RandomVariable object

        print((x_rv+1)**2)  # returns a RandomVariable object with operation recorded

    Defining Bayesian problem using algebra on distributions (creating random variables implicitly):

    .. code-block:: python

        from cuqi.testproblem import Deconvolution1D
        from cuqi.distribution import Gaussian, Gamma, GMRF
        from cuqi.problem import BayesianProblem
        import numpy as np
        A, y_obs, info = Deconvolution1D.get_components()

        # Bayesian problem
        d = Gamma(1, 1e-4)
        s = Gamma(1, 1e-4)
        x = GMRF(np.zeros(A.domain_dim), d)
        y = Gaussian(A @ x, 1/s)

        BP = BayesianProblem(y, x, s, d)
        BP.set_data(y=y_obs)
        BP.UQ()

    """

    def __init__(self, distribution, operations=None, operations_string=None):
        """ Create random variable from distribution """
        if distribution.name is None or distribution.name == "distribution":
            raise ValueError(
                "Unable to create random variable from distribution without name. Ensure "
                "distribution is defined as a variable, e.g. x = Gaussian(0, 1) or provide "
                "a name, e.g. Gaussian(0, 1, name='x')"
            )
        
        self._distribution = distribution
        """ The distribution from which the random variable originates. """

        self._operations = operations or []
        """ List of operations to apply to random variable. """

        self._operations_string = operations_string
        """ String representation of operations on random variable. """

    def __call__(self, *args, **kwargs) -> Any:
        """ Evaluate random variable at a given parameter value. """
        if args and kwargs:
            raise ValueError("Cannot pass both positional and keyword arguments to RandomVariable")

        if len(args) > 1 or len(kwargs) > 1:
            raise ValueError("Cannot pass more than one argument to RandomVariable")
        
        value = self._parse_args_and_kwargs(*args, **kwargs)

        # Apply operations to value sequentially
        for operation in self._operations:
            value = operation(value)

        return value

    @property
    def operations_string(self) -> str:
        """ String representation of operations on random variable. """
        return self._operations_string or self.parameter_name   
        
    @operations_string.setter
    def operations_string(self, value):
        self._operations_string = value

    @property
    def parameter_name(self) -> str:
        """ Name of the parameter that the random variable can be evaluated at. """
        return self._distribution.name  # type: ignore
    
    @property
    def _non_default_args(self) -> List[str]:
        """List of non-default arguments to distribution. This is used to return the correct
        arguments when evaluating the random variable.
        """
        return [self._distribution.name] # type: ignore
    
    def _parse_args_and_kwargs(self, *args, **kwargs):
        """ Parse args and kwargs to get input value for random variable. """
        if args:
            return args[0]
        
        for key, value in kwargs.items():
            if key in self._non_default_args:
                return value
            
        expected_param = self._non_default_args[0]
        raise ValueError(f"Must pass either a positional or keyword argument that matches the expected parameter: {expected_param}")

    def __repr__(self):
        return (f"RandomVariable\n"
                f"Original: {self.parameter_name} ~ {self._distribution}\n"
                f"Transformations: {self.operations_string}")

    def _lazy_apply_operation(self, operation, operation_string_lambda):
        """
        Apply a new operation to this random variable, returning a new random variable.

        This function is "lazy" in that it doesn't actually perform the operation,
        but instead records it in a list of operations to be applied later.
        
        The new random variable will share the same underlying distribution, but will have 
        a modified list of operations and a modified string representation reflecting 
        the new operation.

        Args:
            operation (callable): A function that represents the new operation. This function 
                should take one argument (the value to be operated on) and return the result 
                of the operation.

            operation_string_lambda (callable): A function that takes the current operation 
                string and returns the new operation string. This function should reflect 
                the new operation being added.

        Returns:
            RandomVariable: A new random variable with the added operation.
        """
        new_operations = self._operations + [operation]
        new_operations_string = operation_string_lambda(self.operations_string)
        return RandomVariable(self._distribution, new_operations, new_operations_string)

    # Following methods redefine arithmetic operations for the RandomVariable. 
    # They return a new RandomVariable with the respective operation added to its transformation list.
    
    def __add__(self, other) -> RandomVariable:
        return self._lazy_apply_operation(lambda x: x + other, lambda s: f"{s} + {other}")

    def __radd__(self, other) -> RandomVariable:
        return self.__add__(other)

    def __sub__(self, other) -> RandomVariable:
        return self._lazy_apply_operation(lambda x: x - other, lambda s: f"{s} - {other}")

    def __rsub__(self, other) -> RandomVariable:
        return self.__sub__(other)

    def __mul__(self, other) -> RandomVariable:
        return self._lazy_apply_operation(lambda x: x * other, lambda s: f"({s}) * {other}")

    def __rmul__(self, other) -> RandomVariable:
        return self.__mul__(other)

    def __truediv__(self, other) -> RandomVariable:
        return self._lazy_apply_operation(lambda x: x / other, lambda s: f"({s}) / {other}")

    def __rtruediv__(self, other) -> RandomVariable:
        return self._lazy_apply_operation(lambda x: other / x, lambda s: f"{other} / ({s})")

    def __matmul__(self, other) -> RandomVariable:
        return self._lazy_apply_operation(lambda x: x @ other, lambda s: f"({s}) @ {other}")

    def __rmatmul__(self, other) -> RandomVariable:
        return self._lazy_apply_operation(lambda x: other @ x, lambda s: f"{other} @ ({s})")

    def __neg__(self) -> RandomVariable:
        return self._lazy_apply_operation(lambda x: -x, lambda s: f"-({s})")

    def __abs__(self) -> RandomVariable:
        return self._lazy_apply_operation(lambda x: abs(x), lambda s: f"abs({s})")

    def __pow__(self, other) -> RandomVariable:
        return self._lazy_apply_operation(lambda x: x ** other, lambda s: f"({s}) ^ {other}")

    def __getitem__(self, i) -> RandomVariable:
        return self._lazy_apply_operation(lambda x: x[i], lambda s: f"({s})[{i}]")
