from cuqi.utilities import get_non_default_args
from cuqi.distribution import Distribution, Gaussian, GMRF
from cuqi.solver import ProjectNonnegative, ProjectBox, ProximalL1

import numpy as np


class ImplicitRegularizedGaussian(Distribution):
    def __init__(self, gaussian, proximal = None, projector = None, constraint = None, regularization = None, **kwargs):
        # Underlying explicity Gaussian
        self._gaussian = gaussian
        
        super().__init__(geometry = gaussian.geometry, **kwargs) 
        
        if (proximal is not None) + (projector is not None) + (constraint is not None) + (regularization is not None) != 1:
            raise ValueError("Incorrect parameters")
        # Init from abstract distribution class

        if proximal is not None:
            if not callable(proximal):
                raise ValueError("Proximal needs to be callable.")
            if len(get_non_default_args(proximal)) != 2:
                raise ValueError("Proximal should take 2 arguments.")
            
        if projector is not None:
            if not callable(projector):
                raise ValueError("Projector needs to be callable.")
            if len(get_non_default_args(projector)) != 1:
                raise ValueError("Projector should take 1 argument.")
                
        if gaussian is not None and not isinstance(gaussian, (Gaussian, GMRF)):
            raise ValueError("Explicit underlying distribution needs to be a gaussian ")
            
        # Preset information, for use in Gibbs
        self._preset = None
        
        if proximal is not None:
            self._proximal = proximal
        elif projector is not None:
            self._proximal = lambda z, gamma: projector(z)
        elif (isinstance(constraint, str) and constraint.lower() in ["nonnegativity", "nonnegative", "nn"]):
            self._proximal = lambda z, gamma: ProjectNonnegative(z)
            self._preset = "nonnegativity"
        elif (isinstance(constraint, str) and constraint.lower() in ["box"]):
            self._proximal = lambda z, gamma: ProjectBox(z)
            self._preset = "box" # Not supported in Gibbs
        elif (isinstance(regularization, str) and regularization.lower() in ["l1"]):
            self._proximal = ProximalL1
            self._preset = "l1"
        else:
            raise ValueError("Regularization not supported")
            

    # This is a getter only attribute for the underlying Gaussian
    # It also ensures that the name of the underlying Gaussian
    # matches the name of the implicit regularized Gaussian
    @property
    def gaussian(self):
        if self._name is not None:
            self._gaussian._name = self._name
        return self._gaussian
    
    @property
    def proximal(self):
        return self._proximal
    
    @property
    def preset(self):
        return self._preset

    def logpdf(self, x):
        return np.nan
        #raise ValueError(
        #    f"The logpdf of a implicit regularized Gaussian distribution need not be defined.")
        
    def _sample(self, N, rng=None):
        raise ValueError(
            "There is no known way of efficiently sampling from a implicit regularized Gaussian distribution need not be defined.")
  

    # --- Defer behavior of the underlying Gaussian --- #
    @property
    def geometry(self):
        return self.gaussian.geometry
    
    @geometry.setter
    def geometry(self, value):
        self._gaussian.geometry = value
    
    @property
    def mean(self):
        return self.gaussian.mean
    
    @mean.setter
    def mean(self, value):
        self.gaussian.mean = value
    
    @property
    def cov(self):
        return self.gaussian.cov
    
    @cov.setter
    def cov(self, value):
        self.gaussian.cov = value
    
    @property
    def prec(self):
        return self.gaussian.prec
    
    @prec.setter
    def prec(self, value):
        self.gaussian.prec = value
    
    @property
    def sqrtprec(self):
        return self.gaussian.sqrtprec
    
    @sqrtprec.setter
    def sqrtprec(self, value):
        self.gaussian.sqrtprec = value
    
    @property
    def sqrtcov(self):
        return self.gaussian.sqrtcov
    
    @sqrtcov.setter
    def sqrtcov(self, value):
        self.gaussian.sqrtcov = value     
    
    def get_conditioning_variables(self):
        return self.gaussian.get_conditioning_variables()
    
    def get_mutable_variables(self):
        return self.gaussian.get_mutable_variables()
    
    # Overwrite the condition method such that the underlying Gaussian is conditioned in general, except when conditioning on self.name
    # which means we convert Distribution to Likelihood or EvaluatedDensity.
    def _condition(self, *args, **kwargs):

        # Handle positional arguments (similar code as in Distribution._condition)
        cond_vars = self.get_conditioning_variables()
        kwargs = self._parse_args_add_to_kwargs(cond_vars, *args, **kwargs)

        # When conditioning, we always do it on a copy to avoid unintentional side effects
        new_density = self._make_copy()

        # Check if self.name is in the provided keyword arguments.
        # If so, pop it and store its value.
        value = kwargs.pop(self.name, None)

        new_density._gaussian = self.gaussian._condition(**kwargs)

        # If self.name was provided, we convert to a likelihood or evaluated density
        if value is not None:
            new_density = new_density.to_likelihood(value)

        return new_density