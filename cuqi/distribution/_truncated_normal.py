import numpy as np
from scipy.special import erf
from cuqi.distribution import Distribution

class TruncatedNormal(Distribution):
    """
    Truncated Normal probability distribution. Generates instance of cuqi.distribution.TruncatedNormal.
    
    The variables of this distribution are iid.

    
    Parameters
    ------------
    mean: mean of distribution
    std: standard deviation
    a: lower bound of the distribution
    b: upper bound of the distribution
    """
    def __init__(self, mean=None, std=None, a=None, b=None, is_symmetric=False, **kwargs):
        # Init from abstract distribution class
        super().__init__(is_symmetric=is_symmetric, **kwargs)  

        # Init specific to this distribution
        self.mean = mean
        self.std = std
        self.a = a
        self.b = b

    def logpdf(self, x):
        # the unnormalized logpdf
        # check if x falls in the range between np.array a and b
        if np.any(x < self.a) or np.any(x > self.b):
            return -np.Inf
        else:
             return np.sum(-np.log(self.std*np.sqrt(2*np.pi))-0.5*((x-self.mean)/self.std)**2)

    def gradient(self, x):
        # check if x falls in the range between np.array a and b
        if np.any(x < self.a) or np.any(x > self.b):
            return np.NaN*np.ones_like(x)
        else:
            return -(x-self.mean)/(self.std**2)

    def _sample(self,N=1, rng=None):
        pass