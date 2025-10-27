from __future__ import annotations
from typing import List, Any, Union
from ._ast import VariableNode, Node
from ._orderedset import _OrderedSet
import operator
import cuqi
from cuqi.distribution import Distribution
from copy import copy, deepcopy
import numpy as np


class RandomVariable:
    """ Random variable defined by a distribution with the option to apply algebraic operations on it.

    Random variables allow for the definition of Bayesian Problems in a natural way. In the context
    of code, the random variable can be viewed as a lazily evaluated variable/array. It records
    operations applied to it and acts as a function that, when called, evaluates the operations
    and returns the result.

    In CUQIpy, random variables can be in two forms: (1) a 'primal' random variable that is directly
    defined by a distribution, e.g. x ~ N(0, 1), or (2) a 'transformed' random variable that is defined by
    applying algebraic operations on one or more random variables, e.g. y = x + 1.

    This distinction is purely for the purpose of the implementation in CUQIpy, as mathematically both
    x ~ N(0, 1) and y = x + 1 ~ N(1, 1) are random variables. The distinction is useful for the
    code implementation. In the future some operations like the above may allow primal random variables
    that are transformed if the distribution can be analytically described.

    Parameters
    ----------
    distributions : Distribution or list of Distributions
        The distribution from which the random variable originates. If multiple distributions are
        provided, the random variable is defined by the passed abstract syntax `tree` representing the
        algebraic operations applied to one or more random variables.

    tree : Node, optional
        The tree, represented by the syntax tree nodes, that contain the algebraic operations applied to the random variable.
        Specifically, the root of the tree should be provided.

    name : str, optional
        Name of the random variable. If not provided, the name is extracted from either the distribution provided
        or from the variable name in the code. The name provided must match the parameter name of the distribution.

    Example
    -------

    Basic usage:

    .. code-block:: python

        from cuqi.distribution import Gaussian

        x = RandomVariable(Gaussian(0, 1))

    Defining Bayesian problem using random variables:

    .. code-block:: python

        from cuqi.testproblem import Deconvolution1D
        from cuqi.distribution import Gaussian, Gamma, GMRF
        from cuqi.experimental.algebra import RandomVariable
        from cuqi.problem import BayesianProblem

        import numpy as np
        A, y_obs, info = Deconvolution1D().get_components()

        # Bayesian problem
        d = RandomVariable(Gamma(1, 1e-4))
        s = RandomVariable(Gamma(1, 1e-4))
        x = RandomVariable(GMRF(np.zeros(A.domain_dim), d))
        y = RandomVariable(Gaussian(A @ x, 1/s))

        BP = BayesianProblem(y, x, s, d)
        BP.set_data(y=y_obs)
        BP.UQ()

    Defining random variable from multiple distributions:

    .. code-block:: python

        from cuqi.distribution import Gaussian, Gamma
        from cuqi.experimental.algebra import RandomVariable, VariableNode

        # Define the variables
        x = VariableNode('x')
        y = VariableNode('y')

        # Define the distributions (names must match variables)
        dist_x = Gaussian(0, 1, name='x')
        dist_y = Gamma(1, 1e-4, name='y')

        # Define the tree (this is the algebra that defines the random variable along with the distributions)
        tree = x + y

        # Define random variable from 2 distributions with relation x+y
        rv = RandomVariable([dist_x, dist_y], tree)

    """


    def __init__(self, distributions: Union['Distribution', List['Distribution']], tree: 'Node' = None, name: str = None):
        """ Create random variable from distribution """

        if isinstance(distributions, Distribution):
            distributions = [distributions]
        
        if not  isinstance(distributions, list) and not isinstance(distributions, _OrderedSet):
            raise ValueError("Expected a distribution or a list of distributions")

        # Convert single distribution(s) to internal datastructure _OrderedSet.
        # We use ordered set to ensure that the order of the distributions is preserved.
        # which in turn ensures that the parameter names are always in the same order.
        if not isinstance(distributions, _OrderedSet):
            distributions = _OrderedSet(distributions)

        # If tree is provided, check it is consistent with the given distributions
        if tree:
            tree_var_names = tree.get_variables()
            dist_par_names = {dist._name for dist in distributions}

            if len(tree_var_names) != len(distributions):
                raise ValueError(
                    f"There are {len(tree_var_names)} variables in the tree, but {len(distributions)} distributions are provided. "
                    "This may be due to passing multiple distributions with the same parameter name. "
                    f"The tree variables are {tree_var_names} and the distribution parameter names are {dist_par_names}."
                )
            
            if not all(var_name in dist_par_names for var_name in tree_var_names):
                raise ValueError(
                    f"Variable names in the tree {tree_var_names} do not match the parameter names in the distributions {dist_par_names}. "
                    "Ensure the name is inferred from the variable or explicitly provide it using name='var_name' in the distribution."
                )

        # Match random variable name with distribution parameter name (for single distribution)
        if len(distributions) == 1 and tree is None:
            dist = next(iter(distributions))
            dist_par_name = dist._name
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
        """ Name of the random variable. """


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

    def sample(self, N=1):
        """ Sample from the random variable. 

        Parameters
        ----------
        N : int, optional
            Number of samples to draw. Default is 1.
        """

        if self.is_cond:
            raise NotImplementedError(
                "Unable to directly sample from a random variable that has distributions with "
                "conditioning variables. This is not implemented."
            )

        if N == 1: return self(**{dist.name: dist.sample() for dist in self.distributions})
            
        samples = np.array([
            self(**{dist.name: dist.sample() for dist in self.distributions})
            for _ in range(N)
        ]).reshape(-1, N)  # Ensure correct shape (dim, N)

        return cuqi.samples.Samples(samples)
    
    @property
    def tree(self):
        if self._tree is None:
            if len(self._distributions) > 1:
                raise ValueError("Tree for multiple distributions can not be created automatically and need to be passed as an argument to the {} initializer.".format(type(self).__name__))
            self._tree = VariableNode(self.name)
        return self._tree

    @property
    def name(self):
        """ Name of the random variable. If not provided, the name is extracted from the variable name in the code. """
        if self._is_copy: # Extract the original variable name if this is a copy
            return self._original_variable.name
        if self._name is None: # If None extract the name from the stack
            self._name = cuqi.utilities._get_python_variable_name(self)
        if self._name is not None:
            self._inject_name_into_distribution(self._name)
        return self._name
    
    @name.setter
    def name(self, name):
        if self._is_copy:
            raise ValueError("This random variable is derived from the conditional random variable named "+self._original_variable.name+". The name of the derived random variable cannot be set, but follows the name of the original random variable.")
        self._name = name
        
    @property
    def distribution(self) -> cuqi.distribution.Distribution:
        """ Distribution from which the random variable originates. """
        if len(self._distributions) > 1:
            raise ValueError("Cannot get distribution from random variable defined by multiple distributions")
        self._inject_name_into_distribution()
        return next(iter(self._distributions))
    
    @property
    def distributions(self) -> set:
        """ Distributions from which the random variable originates. """
        self._inject_name_into_distribution()
        return self._distributions
        
    @property
    def parameter_names(self) -> str:
        """ Name of the parameter that the random variable can be evaluated at. """
        self._inject_name_into_distribution()
        return [distribution._name for distribution in self.distributions] # Consider renaming .name to .par_name for distributions

    @property
    def dim(self):
        if self.is_transformed:
            raise NotImplementedError("Dimension not implemented for transformed random variables")
        return self.distribution.dim

    @property
    def geometry(self):
        if self.is_transformed:
            raise NotImplementedError("Geometry not implemented for transformed random variables")
        return self.distribution.geometry

    @geometry.setter
    def geometry(self, geometry):
        if self.is_transformed:
            raise NotImplementedError("Geometry not implemented for transformed random variables")
        self.distribution.geometry = geometry

    @property
    def expression(self):
        """ Expression (formula) of the random variable. """
        return str(self.tree)

    @property
    def is_transformed(self):
        """ Returns True if the random variable is transformed. """
        return not isinstance(self.tree, VariableNode)

    @property
    def is_cond(self):
        """ Returns True if the random variable is a conditional random variable. """
        return any(dist.is_cond for dist in self.distributions)

    def condition(self, *args, **kwargs):
        """Condition the random variable on a given value. Only one of either positional or keyword arguments can be passed.
        
        Parameters
        ----------
        *args : Any
            Positional arguments to condition the random variable on. The order of the arguments must match the order of the parameter names.

        **kwargs : Any
            Keyword arguments to condition the random variable on. The keys must match the parameter names.
        
        """

        # Before conditioning, capture repr to ensure all variable names are injected
        self.__repr__()

        if args and kwargs:
            raise ValueError("Cannot pass both positional and keyword arguments to RandomVariable")
        
        if args:
            kwargs = self._parse_args_add_to_kwargs(args, kwargs)

        # Create a deep copy of the random variable to ensure the original tree is not modified
        new_variable = self._make_copy(deep=True)

        for kwargs_name in list(kwargs.keys()):
            value = kwargs.pop(kwargs_name)

            # Condition the tree turning the variable into a constant
            if kwargs_name in self.parameter_names:
                new_variable._tree = new_variable.tree.condition(**{kwargs_name: value})
            
            # Condition the random variable on both the distribution parameter name and distribution conditioning variables
            for dist in self.distributions:
                if kwargs_name == dist.name:
                    new_variable._remove_distribution(dist.name)
                elif kwargs_name in dist.get_conditioning_variables():
                    new_variable._replace_distribution(dist.name, dist(**{kwargs_name: value}))

        # Check if any kwargs are left unprocessed
        if kwargs:
            raise ValueError(f"Conditioning variables {list(kwargs.keys())} not found in the random variable {self}")

        return new_variable

    @property
    def _non_default_args(self) -> List[str]:
        """List of non-default arguments to distribution. This is used to return the correct
        arguments when evaluating the random variable.
        """
        return self.parameter_names

    def _replace_distribution(self, name, new_distribution):
        """ Replace distribution with a given name with a new distribution in the same position of the ordered set. """
        for dist in self.distributions:
            if dist._name == name:
                self._distributions.replace(dist, new_distribution)
                break

    def _remove_distribution(self, name):
        """ Remove distribution with a given name from the set of distributions. """
        for dist in self.distributions:
            if dist._name == name:
                self._distributions.remove(dist)
                break

    def _inject_name_into_distribution(self, name=None):
        if len(self._distributions) == 1:
            dist = next(iter(self._distributions))

            if dist._is_copy:
                dist = dist._original_density

            if dist._name is None:
                if name is None:
                    name = self.name
                dist.name = name # Inject using setter
    
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
                f"Expression: {self.tree}\n"
                f"Components: \n{parameter_strings}"
                )
        else:
            body = parameter_strings.replace("\t","")
        return title+body
   
    @property
    def _is_copy(self):
        """ Returns True if this is a copy of another random variable, e.g. by conditioning. """
        return hasattr(self, '_original_variable') and self._original_variable is not None

    def _make_copy(self, deep=False) -> 'RandomVariable':
        """ Returns a copy of the density keeping a pointer to the original. """
        if deep:
            new_variable = deepcopy(self)
            new_variable._original_variable = self
            return new_variable
        new_variable = copy(self)
        new_variable._distributions = copy(self.distributions)
        new_variable._tree = copy(self._tree)
        new_variable._original_variable = self
        return new_variable
    
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
