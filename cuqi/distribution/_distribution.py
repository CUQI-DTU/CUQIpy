from __future__ import annotations
from typing import Union
from abc import ABC, abstractmethod
from copy import copy
from functools import partial
from cuqi.density import Density, EvaluatedDensity
from cuqi.likelihood import Likelihood
from cuqi.samples import Samples, CUQIarray
from cuqi.geometry import _DefaultGeometry, Geometry
from cuqi.utilities import infer_len, get_writeable_attributes, get_writeable_properties, get_non_default_args, get_indirect_variables
import numpy as np # To be replaced by cuqi.array_api

# ========== Abstract distribution class ===========
class Distribution(Density, ABC):
    """ Abstract Base Class for Distributions.

    Handles functionality for pdf evaluation, sampling, geometries and conditioning.
    
    Parameters
    ----------
    name : str, default None
        Name of distribution.
    
    geometry : Geometry, default _DefaultGeometry (or None)
        Geometry of distribution.

    is_symmetric : bool, default None
        Indicator if distribution is symmetric.

    Attributes
    ----------
    dim : int or None
        Dimension of distribution.

    name : str or None
        Name of distribution.
    
    geometry : Geometry or None
        Geometry of distribution.

    is_cond : bool
        Indicator if distribution is conditional.

    Methods
    -------
    pdf():
        Evaluate the probability density function.

    logpdf():
        Evaluate the log probability density function.

    sample():
        Generate one or more random samples.

    get_conditioning_variables():
        Return the conditioning variables of distribution.

    get_mutable_variables():
        Return the mutable variables (attributes and properties) of distribution.

    Notes
    -----
    A distribution can be conditional if one or more mutable variables are unspecified.
    A mutable variable can be unspecified in one of two ways:

    1. The variable is set to None.
    2. The variable is set to a callable function with non-default arguments.

    The conditioning variables of a conditional distribution are then defined to be the
    mutable variable itself (in case 1) or the parameters to the callable function (in case 2).

    """
    def __init__(self, name=None, geometry=None, is_symmetric=None):
        """ Initialize the core properties of the distribution.
        
        Parameters
        ----------
        name : str, default None
            Name of distribution.

        geometry : Geometry, default _DefaultGeometry (or None)
            Geometry of distribution.

        is_symmetric : bool, default None
            Indicator if distribution is symmetric.
                        
        """
        super().__init__(name=name)
        self.is_symmetric = is_symmetric
        self.geometry = geometry

    @property
    def dim(self):
        """ Return the dimension of the distribution.
        
        The dimension is automatically inferred from the mutable variables of the distribution.

        If the dimension can not be inferred, None is returned.

        Subclassing distributions can choose to overwrite this property if different behavior is desired.
        """

        # Get all mutable variables
        mutable_vars = self.get_mutable_variables()

        # Loop over mutable variables and get range dimension of each and get the maximum
        max_len = max([infer_len(getattr(self, var)) for var in mutable_vars])

        return max_len if max_len > 0 else None

    @property
    def geometry(self):
        if self.dim != self._geometry.par_dim:
            if isinstance(self._geometry,_DefaultGeometry):
                self.geometry = self.dim
            else:
                raise Exception("Distribution Geometry attribute is not consistent with the distribution dimension ('dim')")
        # Check if dist has a name, if so we provide it to the geometry
        # We do not use self.name to potentially infer it from python stack.
        if self._name is not None: 
            self._geometry._variable_name = self._name
        return self._geometry

    @geometry.setter
    def geometry(self,value):
        if isinstance(value, (int,np.integer)) or value is None:
            self._geometry = _DefaultGeometry(grid=value)
        elif isinstance(value, Geometry):
            self._geometry = value
        else:
            raise TypeError("The attribute 'geometry' should be of type 'int' or 'cuqi.geometry.Geometry', or None.")

    def logd(self, *args, **kwargs):
        """  Evaluate the un-normalized log density function of the distribution.

        The log density function is equal to the log probability density function (logpdf) of a distribution
        plus an additive constant.

        It is possible to pass conditioning variables as arguments to this function in addition to the parameters of the distribution.

        All distributions are required to implement an un-normalized log density function, but not required
        implement the log probability density function (logpdf). For MCMC sampling, the log density function is
        used to sample from the distribution for efficient sampling.
        
        """

        # Get the (potential) conditioning variables
        cond_vars = self.get_conditioning_variables()

        # If distribution is conditional, we first condition before evaluating the log density
        if len(cond_vars) > 0:

            kwargs = self._parse_args_add_to_kwargs(cond_vars, *args, **kwargs)

            # Check if all conditioning variables are specified
            if not all([key in kwargs for key in cond_vars]):
                raise ValueError(f"{self.logd.__qualname__}: To evaluate the log density all conditioning variables must be specified. Conditioning variables are: {cond_vars}")

            # Extract exactly the conditioning variables from kwargs
            cond_kwargs = {key: kwargs[key] for key in cond_vars}

            # Condition the distribution on the conditioning variables
            new_dist = self(**cond_kwargs)

            # Evaluate the log density of the conditioned distribution
            # We use _main_parameter to avoid extracting the name if not necessary
            if "_main_parameter" in kwargs:
                return new_dist.logd(kwargs["_main_parameter"])
            else:
                main_params = {key: kwargs[key] for key in kwargs if key not in cond_vars}
                return new_dist.logd(**main_params)

        # Not conditional distribution, simply evaluate log density directly
        else:
            return super().logd(*args, **kwargs)
        
    def _logd(self, *args):
        return self.logpdf(*args) # Currently all distributions implement logpdf so we simply call this method.

    @abstractmethod
    def logpdf(self,x):
        """ Evaluate the log probability density function of the distribution.
        
        If the logpdf is only needed for MCMC sampling, consider using the un-normalized
        log density function :meth:`logd` instead of the logpdf for efficiency.
        
        """
        pass

    def sample(self,N=1,*args,**kwargs):
        #Make sure all values are specified, if not give error
        #for key, value in vars(self).items():
        #    if isinstance(value,Distribution) or callable(value):
        #        raise NotImplementedError("Parameter {} is {}. Parameter must be a fixed value.".format(key,value))

        # Get samples from the distribution sample method
        s = self._sample(N,*args,**kwargs)

        #Store samples in cuqi samples object if more than 1 sample
        if N==1:
            if len(s) == 1 and isinstance(s,np.ndarray): #Extract single value from numpy array
                s = s.ravel()[0]
            else:
                s = s.flatten()
            s = CUQIarray(s, geometry=self.geometry)
        else:
            s = Samples(s, self.geometry)

        return s

    @abstractmethod
    def _sample(self,N):
        pass

    def pdf(self,x):
        return np.exp(self.logpdf(x))

    def _condition(self, *args, **kwargs):
        """ Generate new distribution conditioned on the input arguments.
        
        Positional arguments must follow the order of the parameter names of the distribution.
        These can be accessed via the :meth:`get_parameter_names` method.
        
        """

        # Store conditioning variables and mutable variables
        cond_vars = self.get_conditioning_variables()
        mutable_vars = self.get_mutable_variables()

        # We allow the use of positional arguments following the order of the parameter names
        kwargs = self._parse_args_add_to_kwargs(cond_vars, *args, **kwargs)

        # EVALUATE CONDITIONAL DISTRIBUTION
        new_dist = self._make_copy() #New cuqi distribution conditioned on the kwargs
        processed_kwargs = set() # Keep track of processed (unique) elements in kwargs

        # Go through every mutable variable and assign value from kwargs if present
        for var_key in mutable_vars:

            #If keyword directly specifies new value of variable we simply reassign
            if var_key in kwargs:
                setattr(new_dist, var_key, kwargs.get(var_key))
                processed_kwargs.add(var_key)

            # If variable is callable we check if any keyword arguments
            # can be used as arguments to the callable method.
            var_val = getattr(self, var_key) # Get current value of variable
            if callable(var_val):

                accepted_keywords = get_non_default_args(var_val)
                remaining_keywords = copy(accepted_keywords)

                # Builds dict with arguments to call variable with
                var_args = {}
                for kw_key, kw_val in kwargs.items():
                    if kw_key in accepted_keywords:
                        var_args[kw_key] = kw_val
                        remaining_keywords.remove(kw_key)

                # If any keywords matched we evaluate callable variable
                if len(var_args)==len(accepted_keywords):  #All keywords found
                    # Define variable as the output of callable function
                    setattr(new_dist, var_key, var_val(**var_args))

                elif len(var_args)>0:                      #Some keywords found
                    # Define new partial function with partially defined args
                    func = partial(var_val, **var_args)
                    setattr(new_dist, var_key, func)
                
                # Store processed keywords
                processed_kwargs.update(var_args.keys())

        # Check if _main_parameter is specified in kwargs
        # (This is added by the _parse_args_add_to_kwargs method)
        # If so we convert to likelihood with that parameter.
        if "_main_parameter" in kwargs:
            return new_dist.to_likelihood(kwargs["_main_parameter"])

        # Check if any keywords were not used
        unused_kwargs = set(kwargs.keys()) - processed_kwargs

        # If any keywords were not used we must check name.
        # We defer the checking of name to here since it
        # can be slow to automatically determine the name
        # of a distribution by walking the python stack.
        if len(unused_kwargs)>0:

            if self.name in kwargs:
                # If name matches we convert to likelihood
                return new_dist.to_likelihood(kwargs[self.name])  
            else:
                # KEYWORD ERROR CHECK
                for kw_key in kwargs.keys():
                    if kw_key not in (mutable_vars+cond_vars+[self.name]):
                        raise ValueError("The keyword \"{}\" is not a mutable, conditioning variable or parameter name of this distribution.".format(kw_key))       

        return new_dist

    # Overload parent to add type hint.
    def __call__(self, *args, **kwargs) -> Union[Distribution, Likelihood, EvaluatedDensity]:
        return super().__call__(*args, **kwargs)

    def get_conditioning_variables(self):
        """Return the conditioning variables of this distribution (if any)."""
        
        # Get all mutable variables
        mutable_vars = self.get_mutable_variables()

        # Loop over mutable variables and if None they are conditioning variables
        cond_vars = [key for key in mutable_vars if getattr(self, key) is None]

        # Add any variables defined through callable functions
        cond_vars += get_indirect_variables(self)
        
        return cond_vars

    def get_mutable_variables(self):
        """Return any public variable that is mutable (attribute or property) except those in the ignore_vars list"""

        # If mutable variables are already cached, return them
        if hasattr(self, '_mutable_vars'):
            return self._mutable_vars
        
        # Define list of ignored attributes and properties
        ignore_vars = ['name', 'is_symmetric', 'geometry', 'dim']
        
        # Get public attributes
        attributes = get_writeable_attributes(self)

        # Get "public" properties (getter+setter)
        properties = get_writeable_properties(self)

        # Cache the mutable variables
        self._mutable_vars = [var for var in (attributes+properties) if var not in ignore_vars]

        return self._mutable_vars

    def get_parameter_names(self):
        return self.get_conditioning_variables() + [self.name]

    @property
    def is_cond(self):
        """ Returns True if instance (self) is a conditional distribution. """
        if len(self.get_conditioning_variables()) == 0:
            return False
        else:
            return True

    def to_likelihood(self, data):
        """Convert conditional distribution to a likelihood function given observed data"""
        if not self.is_cond: # If not conditional we create a constant density
            return EvaluatedDensity(self.logd(data), name=self.name)
        return Likelihood(self, data)

    def _parse_args_add_to_kwargs(self, cond_vars, *args, **kwargs):
        """ Parse args and add to kwargs. The args are assumed to follow the order of the parameter names.
        
        This particular implementation avoids accessing .get_parameter_names() for speed and requires cond_vars to be passed.
        
        """
        if len(args)>0:
            if len(args) > len(cond_vars)+1:
                raise ValueError(f"{self._condition.__qualname__}: Unable to parse {len(args)} arguments. Only {len(cond_vars)+1} allowed (conditioning variables + main parameter). Use keywords to modify mutable variables.")
            ordered_keys = copy(cond_vars) # Args follow order of cond. vars
            ordered_keys.append("_main_parameter") # Last arg is main parameter
            for index, arg in enumerate(args):
                if index < len(ordered_keys):
                    if ordered_keys[index] in kwargs:
                        raise ValueError(f"{self._condition.__qualname__}: {ordered_keys[index]} passed as both argument and keyword argument.\nArguments follow the listed conditioning variable order: {self.get_conditioning_variables()}")
                    kwargs[ordered_keys[index]] = arg
        return kwargs

    def __repr__(self) -> str:
        if self.is_cond is True:
            return "CUQI {}. Conditioning variables {}.".format(self.__class__.__name__,self.get_conditioning_variables())
        else:
            return "CUQI {}.".format(self.__class__.__name__)
