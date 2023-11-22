from __future__ import annotations
from typing import List, Any
from ._ast import RandomVariableNode
from ._orderedset import OrderedSet
import operator
import cuqi
from copy import copy


class RandomVariable:
    """ Random variable defined by a base distribution with potential algebraic operations applied to it.

    The random variable class is still experimental.

    The random variable can be viewed as a lazily evaluated array. This allows for the definition of
    Bayesian models in a natural way.

    Parameters
    ----------
    distributions : Distribution or set of Distribution
        The distribution from which the random variable originates. If multiple distributions are
        provided, the random variable is defined by the passed abstract syntax `tree` representing the
        algebraic operations on the random variables defined by the distributions.

    tree : RandomVariableNode, optional
        The tree representation of the algebraic operations applied to the random variable.

    Example
    -------

    Basic usage:

    .. code-block:: python

        from cuqi.distribution import Gaussian

        x = Gaussian(0, 1).rv # Returns a RandomVariable object

    Defining Bayesian problem using algebra on distributions (creating random variables implicitly):

    .. code-block:: python

        from cuqi.testproblem import Deconvolution1D
        from cuqi.distribution import Gaussian, Gamma, GMRF
        from cuqi.problem import BayesianProblem
        import numpy as np
        A, y_obs, info = Deconvolution1D().get_components()

        # Bayesian problem
        d = Gamma(1, 1e-4).rv
        s = Gamma(1, 1e-4).rv
        x = GMRF(np.zeros(A.domain_dim), d).rv
        y = Gaussian(A @ x, 1/s).rv

        BP = BayesianProblem(y, x, s, d)
        BP.set_data(y=y_obs)
        BP.UQ()

    """

    def __init__(self, distributions: set, tree: RandomVariableNode = None, name: str = None):
        """ Create random variable from distribution """

        # Convert single distribution to OrderedSet.
        # We use ordered set to ensure that the order of the distributions is preserved.
        # which in turn ensures that the parameter names are always in the same order.
        if not isinstance(distributions, OrderedSet):
            distributions = OrderedSet([distributions])

        # Match random variable name with distribution parameter name (for single distribution)
        if len(distributions) == 1 and tree is None:
            dist = next(iter(distributions))
            dist_par_name = dist.par_name
            if dist_par_name is not None:
                if name is not None and dist_par_name != name:
                    raise ValueError(f"Parameter name '{dist_par_name}' of the distribution does not match the input name '{name}' for the random variable.")
                name = dist_par_name
    
        self._distributions = distributions
        """ The distribution from which the random variable originates. """
        
        self._tree = tree
        """ The tree representation of the random variable. """

        self._original_variable = None
        """ Stores the original variable if this is a conditioned copy"""

        self._name = name

    def __call__(self, *args, **kwargs) -> Any:
        """ Evaluate random variable at a given parameter value. For example, for random variable `X`, `X(1)` gives `1` and `(X+1)(1)` gives `2` """
        if args and kwargs:
            raise ValueError("Cannot pass both positional and keyword arguments to RandomVariable")
        
        if args:
            kwargs = self._parse_args_add_to_kwargs(args, kwargs)

        # Check if kwargs match parameter names using a all compare
        if not all([name in kwargs for name in self.parameter_names]) or not all([name in self.parameter_names for name in kwargs]):
            raise ValueError(f"Expected arguments {self.parameter_names}, got arguments {kwargs}")

        return self.tree(**kwargs)
    
    @property
    def tree(self):
        if self._tree is None:
            if len(self._distributions) > 1:
                raise ValueError("Tree for multiple distributions can not be created automatically and need to be passed as an argument to the {} initializer.".format(type(self).__name__))
            self._tree = RandomVariableNode(self.name)
        return self._tree

    @property
    def name(self):
        """ Name of the random variable. """
        if self._is_copy: # Extract the original density name
            return self._original_variable.name
        if self._name is None: # If None extract the name from the stack
            self._name = cuqi.utilities._get_python_variable_name(self)
        if len(self._distributions) == 1: # If single distribution, inject name into distribution
            self._set_dist_name_if_not_set(next(iter(self._distributions)), self._name)
        return self._name
    
    @name.setter
    def name(self, name):
        if self._is_copy:
            raise ValueError("This random variable is derived from the conditional random variable named "+self._original_variable.name+". The name of the derived random variable cannot be set, but follows the name of the original random variable.")
        self._name = name

    @property
    def _is_copy(self):
        """ Returns True if this is a copy of another random variable, e.g. by conditioning. """
        return hasattr(self, '_original_variable') and self._original_variable is not None
    
    def logd(self, *args, **kwargs):
        if len(self._distributions) > 1 or not isinstance(self.tree, RandomVariableNode):
            raise ValueError("Unable to evaluate log density of transformed random variables")
        return self.dist.logd(*args, **kwargs)
    
    @property
    def is_cond(self):
        if self.is_transformed:
            raise NotImplementedError("Conditioning is not implemented for transformed random variables")
        return self.dist.is_cond

    def gradient(self, *args, **kwargs):
        if self.is_transformed:
            raise NotImplementedError("Gradient not implemented for transformed random variables")
        return self.dist.gradient(*args, **kwargs)

    def sample(self):
        """ Sample random variable. """
        return self(**{distribution.par_name: distribution.sample() for distribution in self.distributions})
         
    @property
    def dist(self) -> cuqi.distribution.Distribution:
        """ Distribution from which the random variable originates. """
        if len(self._distributions) > 1:
            raise ValueError("Cannot get distribution from random variable defined by multiple distributions")
        distribution = next(iter(self._distributions))
        # Inject name into distribution
        self._set_dist_name_if_not_set(distribution)
        return distribution
    
    @property
    def distributions(self) -> set:
        """ Distributions from which the random variable originates. """
        # Inject name into distributions
        if len(self._distributions) == 1:
            self._set_dist_name_if_not_set(next(iter(self._distributions)))
        return self._distributions
    
    def _set_dist_name_if_not_set(self, distribution, name=None):
        if distribution.par_name is None:
            if name is None:
                name = self.name
            distribution.par_name = name

    def get_conditioning_variables(self):
        """ Get conditioning variables. """
        if self.is_transformed:
            raise NotImplementedError("Extracting conditioning variables is not implemented for transformed random variables")
        return self.dist.get_conditioning_variables()
    
    @property
    def parameter_names(self) -> str:
        """ Name of the parameter that the random variable can be evaluated at. """
        return [distribution.par_name for distribution in self.distributions]
    
    @property
    def _non_default_args(self) -> List[str]:
        """List of non-default arguments to distribution. This is used to return the correct
        arguments when evaluating the random variable.
        """
        return self.parameter_names
    
    def _parse_args_add_to_kwargs(self, args, kwargs) -> dict:
        """ Parse args and add to kwargs if any. Arguments follow self.parameter_names order. """
        if len(args) != len(self.parameter_names):
            raise ValueError(f"Expected {len(self.parameter_names)} arguments, got {len(args)}. Parameters are: {self.parameter_names}")
        
        # Add args to kwargs
        for arg, name in zip(args, self.parameter_names):
            kwargs[name] = arg
            
        return kwargs

    def __repr__(self):
        # Create strings for parameter name ~ distribution pairs
        parameter_strings = [f"{name} ~ {distribution}" for name, distribution in zip(self.parameter_names, self.distributions)]
        # Join strings with newlines
        parameter_strings = "\n".join(parameter_strings)
        # Add initial newline and indentations
        parameter_strings = "\n".join(["\t"+line for line in parameter_strings.split("\n")])
        # Print parameter strings with newlines
        if self.is_transformed:
            title = f"Transformed Random Variable"
        else:
            title = f""
        if self.is_transformed:
            body = (
                f"\n"
                f"Formula: {self.tree}\n"
                f"Components: \n{parameter_strings}"
                )
        else:
            body = parameter_strings.replace("\t","")
        return title+body
   
    @property
    def dim(self):
        if self.is_transformed:
            raise NotImplementedError("Dimension not implemented for transformed random variables")
        return self.dist.dim

    @property
    def geometry(self):
        if self.is_transformed:
            raise NotImplementedError("Geometry not implemented for transformed random variables")
        return self.dist.geometry

    @geometry.setter
    def geometry(self, geometry):
        if self.is_transformed:
            raise NotImplementedError("Geometry not implemented for transformed random variables")
        self.dist.geometry = geometry

    def condition(self, *args, **kwargs):
        """ Condition random variable on fixed values """
        if self.is_transformed:
            raise NotImplementedError("Conditioning is not implemented for transformed random variables")
        new_variable = self._make_copy()
        conditioned_dist = self.dist(*args, **kwargs)
        new_variable._distributions = OrderedSet([conditioned_dist])
        return new_variable

    def _make_copy(self):
        """ Returns a shallow copy of the density keeping a pointer to the original. """
        new_variable = copy(self)
        new_variable._distributions = copy(self.distributions)
        new_variable._tree = copy(self._tree)
        new_variable._original_variable = self
        return new_variable

    @property
    def is_transformed(self):
        #return len(self.distributions) > 1 or not self._tree is None
        return not isinstance(self.tree, RandomVariableNode)
    
    def _apply_operation(self, operation, other=None) -> 'RandomVariable':
        """
        Apply a specified operation to this RandomVariable.
        """
        if isinstance(other, cuqi.distribution.Distribution):
            raise ValueError("Cannot apply operation to distribution. Use .rv to create random variable first.")
        if other is None: # unary operation case
            return RandomVariable(self.distributions, operation(self.tree))
        elif isinstance(other, RandomVariable): # binary operation case with another random variable that has distributions
            return RandomVariable(self.distributions | other.distributions, operation(self.tree, other.tree))
        return RandomVariable(self.distributions, operation(self.tree, other)) # binary operation case with any other object (constant)

    def __add__(self, other) -> 'RandomVariable':
        return self._apply_operation(operator.add, other)

    def __radd__(self, other) -> 'RandomVariable':
        return self.__add__(other)

    def __sub__(self, other) -> 'RandomVariable':
        return self._apply_operation(operator.sub, other)

    def __rsub__(self, other) -> 'RandomVariable':
        return self._apply_operation(lambda x, y: operator.sub(y, x), other)

    def __mul__(self, other) -> 'RandomVariable':
        return self._apply_operation(operator.mul, other)

    def __rmul__(self, other) -> 'RandomVariable':
        return self.__mul__(other)

    def __truediv__(self, other) -> 'RandomVariable':
        return self._apply_operation(operator.truediv, other)

    def __rtruediv__(self, other) -> 'RandomVariable':
        return self._apply_operation(lambda x, y: operator.truediv(y, x), other)

    def __matmul__(self, other) -> 'RandomVariable':
        if isinstance(other, cuqi.model.Model) and not isinstance(other, cuqi.model.LinearModel):
            raise TypeError("Cannot apply matmul to non-linear models")
        return self._apply_operation(operator.matmul, other)

    def __rmatmul__(self, other) -> 'RandomVariable':
        if isinstance(other, cuqi.model.Model) and not isinstance(other, cuqi.model.LinearModel):
            raise TypeError("Cannot apply matmul to non-linear models")
        return self._apply_operation(lambda x, y: operator.matmul(y, x), other)

    def __neg__(self) -> 'RandomVariable':
        return self._apply_operation(operator.neg)

    def __abs__(self) -> 'RandomVariable':
        return self._apply_operation(abs)

    def __pow__(self, other) -> 'RandomVariable':
        return self._apply_operation(operator.pow, other)

    def __getitem__(self, other) -> 'RandomVariable':
        return self._apply_operation(operator.getitem, other)
