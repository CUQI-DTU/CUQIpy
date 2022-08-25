from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
from copy import copy
import cuqi

class Density(ABC):
    """ Abstract base class for densities.

    Exposes a unified interface for evaluating the log density
    depending on the parameters of the density. Common subclasses
    are Distribution and Likelihood.

    Parameters
    ----------
    name: str
        Name of the random variable associated with the density.

    Notes
    -----
    Subclasses must implement the following:

    - _logd(self, *args, **kwargs). Returns the log density given a set of parameters.
    - _condition(self, *args, **kwargs). This method conditions (fixes) the density on a given set of parameters.
    - dim. The dimension of the density.
    - get_parameter_names(self). Returns a list of the names of the parameters of the density.

    """
    def __init__(self, name: Optional[str] = None):
        if not isinstance(name, str) and name is not None:
            raise ValueError(f"{self.__init__.__qualname__}: Name must be a string or None")
        self.name = name
        self._constant = 0 # Precomputed constant to add to the log probability.
        self._original_density = None # Original density if this is a conditioned copy. Used to extract name.

    @property
    def name(self):
        """ Name of the random variable associated with the density. """
        if self._is_copy: # Extract the original density name
            return self._original_density.name
        if self._name is None: # If None extract the name from the stack
            self._name = cuqi.utilities._get_python_variable_name(self)
        return self._name

    @name.setter
    def name(self, name):
        if self._is_copy:
            raise ValueError("Cannot set name of conditioned density. Only the original density can have its name set.")
        self._name = name

    @property
    def _is_copy(self):
        """ Returns True if this is a copy of another density, e.g. by conditioning. """
        return hasattr(self, '_original_density') and self._original_density is not None

    def logd(self, *args, **kwargs):
        """ Evaluates the un-normalized log density function given a set of parameters.
        
        The arguments to the method should match the parameter names of the density.
        These can be accessed with the :meth:`get_parameter_names` method.
        
        """

        # Check if kwargs are given. If so parse them according to the parameter names and add them to args.
        if len(kwargs) > 0:

            if len(args) > 0:
                raise ValueError(f"{self.logd.__qualname__}: Cannot specify both positional and keyword arguments.")

            # Get parameter names possible to evaluate the logd
            par_names = self.get_parameter_names()

            # Check if parameter names match the keyword arguments (any order).
            if set(par_names) != set(kwargs.keys()):
                raise ValueError(f"{self.logd.__qualname__}: Parameter names {par_names} do not match keyword arguments {kwargs.keys()}.")
            
            # Ensure that the keyword arguments are given in the correct order and use them as positional arguments.
            args = [kwargs[name] for name in par_names]            

        return self._logd(*args) + self._constant
   
    @abstractmethod
    def _logd(self):
        pass

    @abstractmethod
    def _condition(self):
        pass

    @property
    @abstractmethod
    def dim(self):
        pass

    @abstractmethod
    def get_parameter_names(self):
        """ Returns the names of the parameters that the density can be evaluated at or conditioned on. """
        pass

    def _make_copy(self):
        """ Returns a shallow copy of the density keeping a pointer to the original. """
        new_density = copy(self)
        new_density._original_density = self
        return new_density

    def __call__(self, *args, **kwargs):
        """ Condition the density on a set of parameters.
        
        Positional arguments must follow the order of the parameter names.
        These can be accessed via the :meth:`get_parameter_names` method.

        Conditioning maintains the name of the random variable associated with the density.
        
        """
        return self._condition(*args, **kwargs)

class EvaluatedDensity(Density):
    """ An evaluated density representing a constant number exposed through the logd method.

    EvaluatedDensity is a density that has been evaluated for a particular value/observation of the underlying random variable. It simply returns the value of the log density function evaluated at that value.
    The density has a fixed dimension of 1 and cannot be conditioned on any parameters.

    Parameters
    ----------
    value: float
        The fixed scalar value of the log density function evaluated at the particular value of the underlying random variable. This value will be returned if the EvaluatedDensity is queried for its log density function.

    """
    def __init__(self, value, **kwargs):
        super().__init__(**kwargs)
        self.value = value

    def _logd(self):
        return self.value

    def _condition(self, *args, **kwargs):
        return self

    # Overload parent to add type hint.
    def __call__(self, *args, **kwargs) -> EvaluatedDensity:
        return super().__call__(*args, **kwargs)

    @property
    def dim(self):
        return 1

    def get_parameter_names(self):
        return []

    def __repr__(self):
        return f"EvaluatedDensity({self.logd()})"
