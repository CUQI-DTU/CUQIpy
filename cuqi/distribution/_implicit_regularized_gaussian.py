from cuqi.utilities import get_non_default_args
from cuqi.distribution import Distribution, Gaussian
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
            if len(get_non_default_args(proximal) != 2):
                raise ValueError("Proximal should take 2 arguments.")
            
        if projector is not None:
            if not callable(projector):
                raise ValueError("Projector needs to be callable.")
            if len(get_non_default_args(proximal) != 1):
                raise ValueError("Projector should take 1 argument.")
                
        if gaussian is not None and not isinstance(gaussian, Gaussian):
            raise ValueError("Explicit underlying distribution needs to be a gaussian")
        
            
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
            

    def get_explicit_Gaussian(self):
        return self._gaussian
    
    def get_proximal(self):
        return self._proximal
    
    def get_preset(self):
        return self._preset

    def logpdf(self, x):
        return np.nan
        #raise ValueError(
        #    f"The logpdf of a implicit regularized Gaussian distribution need not be defined.")
        
    def _sample(self, N, rng=None):
        raise ValueError(
            "There is no known way of efficiently sampling from a implicit regularized Gaussian distribution need not be defined.")
  
    @property
    def geometry(self):
        return self._gaussian.geometry
    
    @geometry.setter
    def geometry(self, value):
        self._gaussian.geometry = value
        
    