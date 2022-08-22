from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional

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

    def logd(self, *args, **kwargs):
        """ Evaluates the log density given a set of parameters.
        
        The log density can be evaluated with a set of arguments matching the parameter names
        of the density. These can be accessed with the :meth:`get_parameter_names` method.
        """

        # Get parameter names possible to evaluate the logd
        par_names = self.get_parameter_names()

        # Check if kwargs are given. If so parse them according to the parameter names and add them to args.
        if len(kwargs) > 0:

            if len(args) > 0:
                raise ValueError(f"{self.logd.__qualname__}: Cannot specify both positional and keyword arguments.")

            # Check if parameter names match the keyword arguments (any order).
            if set(par_names) != set(kwargs.keys()):
                raise ValueError(f"{self.logd.__qualname__}: Parameter names {par_names} do not match keyword arguments {kwargs.keys()}.")
            
            # Ensure that the keyword arguments are given in the correct order and use them as positional arguments.
            args = [kwargs[name] for name in par_names]            

        # Check if the number of arguments matches the number of parameters.
        if len(args) != len(par_names):
            raise ValueError(f"{self.logd.__qualname__}: Number of arguments must match number of parameters. Got {len(args)} arguments but density has {len(par_names)} parameters.")

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

    def __call__(self, *args, **kwargs):
        """ Condition on the given parameters. """
        return self._condition(*args, **kwargs)

class ConstantDensity(Density):
    """ A constant density.

    Parameters
    ----------
    value: float
        The constant value of the density.
        
    """
    def __init__(self, value, **kwargs):
        super().__init__(**kwargs)
        self.value = value

    def _logd(self):
        return self.value

    def _condition(self, *args, **kwargs):
        return self

    # Overload parent to add type hint.
    def __call__(self, *args, **kwargs) -> ConstantDensity:
        return super().__call__(*args, **kwargs)

    @property
    def dim(self):
        return 1

    def get_parameter_names(self):
        return []

    def __repr__(self):
        return f"ConstantDensity({self.logd()})"
        