import numpy as np
from cuqi.distribution import Distribution
from cuqi.geometry import Geometry

class Uniform(Distribution):


    def __init__(self, low=None, high=None, is_symmetric=True, **kwargs):
        """
        Parameters
        ----------
        low : float or array_like of floats
            Lower bound(s) of the uniform distribution.
        high : float or array_like of floats 
            Upper bound(s) of the uniform distribution.
        """
        # Init from abstract distribution class
        super().__init__(is_symmetric=is_symmetric, **kwargs)       

        # Init specific to this distribution
        self.low = low
        self.high = high      

    def logpdf(self, x):
        """
        Evaluate the logarithm of the PDF at the given values of x.
        """
        # First check whether x is outside bounds.
        # It is outside if any coordinate is outside the interval.
        if np.any(x < self.low) or np.any(x > self.high):
            # If outside always return -inf
            return_val = -np.inf  
        else:
            # If inside, return the (unnormalized) density
            return_val = 1.0
        return return_val

    def gradient(self, x):
        """
        Computes the gradient of logpdf at the given values of x.
        """
        if np.any(x < self.low) or np.any(x > self.high):
            return np.NaN*np.ones_like(x)
        else:
            return np.zeros_like(x)

    def _sample(self,N=1, rng=None):

        if rng is not None:
            s = rng.uniform(self.low, self.high, (N,self.dim)).T
        else:
            s = np.random.uniform(self.low, self.high, (N,self.dim)).T

        return s

class UnboundedUniform(Uniform):
    """
    Unbounded uniform distribution. This is a special case of the
    Uniform distribution, where the lower and upper bounds are set to
    -inf and inf, respectively. This distribution is not normalizable,
    and therefore cannot be sampled from. It is mainly used for
    initializing non-informative priors.
    Parameters
    ----------
    geometry : int or Geometry
        The geometry of the distribution. If an integer is given, it is
        interpreted as the dimension of the distribution. If a
        Geometry object is given, its par_dim attribute is used.
    """
    def __init__(self, geometry, **kwargs):
        if isinstance(geometry, int):
            low = np.full(geometry, -np.inf)
            high = np.full(geometry, np.inf)
        elif isinstance(geometry, Geometry):
            low = np.full(geometry.par_dim, -np.inf)
            high = np.full(geometry.par_dim, np.inf)
        else:
            raise ValueError("geometry must be an integer or a cuqi Geometry")
        super().__init__(low, high, **kwargs)
    def _sample(self, N=1, rng=None):
        raise NotImplementedError("Cannot sample from UnboundedUniform distribution")