from __future__ import annotations
from typing import List, Any
from ._ast import RandomVariableNode
from ._orderedset import OrderedSet
import operator
import cuqi


class RandomVariable:
    """ Random variable defined by a base distribution with potential algebraic operations applied to it.

    The random variable class is still experimental.

    The random variable can be viewed as a lazily evaluated array. This allows for the definition of
    Bayesian models in a natural way.

    Parameters
    ----------
    distributions : Distribution or set of Distribution
        The distribution from which the random variable originates. If multiple distributions are
        provided, the random variable is defined by the abstract syntax tree representing the
        algebraic operations on the distributions that define the random variable.

    tree : RandomVariableNode, optional
        The tree representation of the algebraic operations applied to the random variable.

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

        x_rv = x.rv  # returns a RandomVariable object

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

    def __init__(self, distributions: set, tree: RandomVariableNode = None):
        """ Create random variable from distribution """
        # Convert single distribution to OrderedSet.
        # We use ordered set to ensure that the order of the distributions is preserved.
        # which in turn ensures that the parameter names are always in the same order.
        if not isinstance(distributions, OrderedSet):
            distributions = OrderedSet([distributions])
        
        # Check that distributions have names
        for distribution in distributions:
            if distribution.name is None or distribution.name == "distribution":
                raise ValueError(
                    "Unable to create random variable from distribution without name. Ensure "
                    "distribution is defined as a variable, e.g. x = Gaussian(0, 1) or provide "
                    "a name, e.g. Gaussian(0, 1, name='x')"
                )
        
        self._distributions = distributions
        """ The distribution from which the random variable originates. """

        if tree is None:
            if len(distributions) > 1:
                raise ValueError("Cannot create random variable from multiple distributions")
            tree = RandomVariableNode(next(iter(distributions)).name)
        
        self._tree = tree
        """ The tree representation of the random variable. """

    def __call__(self, *args, **kwargs) -> Any:
        """ Evaluate random variable at a given parameter value. For example, for random variable `X`, `X(1)` gives `1` and `(X+1)(1)` gives `2` """
        if args and kwargs:
            raise ValueError("Cannot pass both positional and keyword arguments to RandomVariable")
        
        if args:
            kwargs = self._parse_args_add_to_kwargs(args, kwargs)

        return self._tree(**kwargs)
    
    def sample(self):
        """ Sample random variable. """
        return self(**{distribution.name: distribution.sample() for distribution in self._distributions})
         
    @property
    def parameter_names(self) -> str:
        """ Name of the parameter that the random variable can be evaluated at. """
        return [distribution.name for distribution in self._distributions]
    
    @property
    def _non_default_args(self) -> List[str]:
        """List of non-default arguments to distribution. This is used to return the correct
        arguments when evaluating the random variable.
        """
        return self.parameter_names
    
    def _parse_args_add_to_kwargs(self, args, kwargs) -> dict:
        """ Parse args and add to kwargs if any. Arguments follow self.parameter_names order. """
        if len(args) != len(self.parameter_names):
            raise ValueError(f"Expected {len(self.parameter_names)} arguments, got {len(args)}")
        
        # Add args to kwargs
        for arg, name in zip(args, self.parameter_names):
            kwargs[name] = arg
        
        return kwargs

    def __repr__(self):
        # Create strings for parameter name ~ distribution pairs
        parameter_strings = [f"{name} ~ {distribution}" for name, distribution in zip(self.parameter_names, self._distributions)]
        # Join strings with newlines
        parameter_strings = "\n".join(parameter_strings)
        # Add initial newline and indentations
        parameter_strings = "\n".join(["\t"+line for line in parameter_strings.split("\n")])
        # Print parameter strings with newlines
        return (f"RandomVariable\n"
                f"Distributions: \n{parameter_strings}\n"
                f"Transformations: {self._tree}")
   
    def _apply_operation(self, operation, other=None) -> 'RandomVariable':
        """
        Apply a specified operation to this RandomVariable.
        """
        if isinstance(other, cuqi.distribution.Distribution):
            other = other._as_random_variable()
        if other is None:
            return RandomVariable(self._distributions, operation(self._tree))
        elif isinstance(other, RandomVariable):
            return RandomVariable(self._distributions | other._distributions, operation(self._tree, other._tree))
        return RandomVariable(self._distributions, operation(self._tree, other))

    def __add__(self, other) -> 'RandomVariable':
        return self._apply_operation(operator.add, other)

    def __radd__(self, other) -> 'RandomVariable':
        return self.__add__(other)

    def __sub__(self, other) -> 'RandomVariable':
        return self._apply_operation(operator.sub, other)

    def __rsub__(self, other) -> 'RandomVariable':
        return self.__sub__(other)

    def __mul__(self, other) -> 'RandomVariable':
        return self._apply_operation(operator.mul, other)

    def __rmul__(self, other) -> 'RandomVariable':
        return self.__mul__(other)

    def __truediv__(self, other) -> 'RandomVariable':
        return self._apply_operation(operator.truediv, other)

    def __rtruediv__(self, other) -> 'RandomVariable':
        return self._apply_operation(lambda x, y: operator.truediv(y, x), other)

    def __matmul__(self, other) -> 'RandomVariable':
        return self._apply_operation(operator.matmul, other)

    def __rmatmul__(self, other) -> 'RandomVariable':
        return self._apply_operation(lambda x, y: operator.matmul(y, x), other)

    def __neg__(self) -> 'RandomVariable':
        return self._apply_operation(operator.neg)

    def __abs__(self) -> 'RandomVariable':
        return self._apply_operation(abs)

    def __pow__(self, other) -> 'RandomVariable':
        return self._apply_operation(operator.pow, other)

    def __getitem__(self, other) -> 'RandomVariable':
        return self._apply_operation(operator.getitem, other)
