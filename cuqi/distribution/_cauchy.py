import numpy as np
import scipy.stats as sps
from cuqi.geometry import _get_identity_geometries
from cuqi.utilities import force_ndarray
from cuqi.distribution import Distribution

class Cauchy(Distribution):
    """ Multivariate Cauchy distribution of independent random variables.
     
    Each is distributed according to the PDF function

    .. math::

        \\frac{1}{\\pi\\gamma(1+\\frac{(x-\\mu)^2}{\\gamma^2})}

    where :math:`\\mu` is the location parameter and :math:`\\gamma` is the scale parameter.

    Parameters
    ----------
    location: float or array_like
        Location parameter

    scale: float or array_like
        Scale parameter

    Example
    -------
    .. code-block:: python

        # % Generate a Cauchy distribution
        import cuqi
        location = 0
        scale  = 1
        x = cuqi.distribution.Cauchy(location, scale)
        x.logpdf(0) # -1.1447298858494002

    .. code-block:: python

        # % Generate a multivariate Cauchy distribution
        import cuqi
        location = [0, 1]
        scale  = [1, 3]
        x = cuqi.distribution.Cauchy(location, scale)
        x.logpdf([0, 0]) # -3.4934325760247367

    """

    def __init__(self, location=None, scale=None, is_symmetric=True, **kwargs):
        super().__init__(is_symmetric=is_symmetric, **kwargs)
        self.location = location
        self.scale = scale

    @property
    def location(self):
        """ Location parameter """
        return self._location
    
    @location.setter
    def location(self, value):
        self._location = force_ndarray(value, flatten=True)

    @property
    def scale(self):
        """ Scale parameter """
        return self._scale
    
    @scale.setter
    def scale(self, value):
        self._scale = force_ndarray(value, flatten=True)

    def _is_out_of_bounds(self, x):
        """ Check if x is out of bounds """
        return np.any(self.scale <= 0)

    def logpdf(self, x):

        if self._is_out_of_bounds(x):
            return -np.inf
        
        return np.sum(-np.log(np.pi*self.scale*(1+((x-self.location)/self.scale)**2)))
    
    def cdf(self, x):

        if self._is_out_of_bounds(x):
            return -np.inf
        
        return np.sum(sps.cauchy.cdf(x, loc=self.location, scale=self.scale))
    
    def gradient(self, x):

        #Avoid complicated geometries that change the gradient.
        if not type(self.geometry) in _get_identity_geometries():
            raise NotImplementedError("Gradient not implemented for distribution {} with geometry {}".format(self,self.geometry))
        
        #Computing the gradient for conditional distribution is not supported yet    
        if self.is_cond:
            raise NotImplementedError(f"Gradient is not implemented for {self} with conditioning variables {self.get_conditioning_variables()}")
        
        # Check bounds (return nan if out of bounds)
        if self._is_out_of_bounds(x):
            return x*np.nan
        
        #Compute the gradient
        x_translated = x-self.location
        return -2*x_translated/(self.scale**2*(1+(x_translated/self.scale)**2))
    
    def _sample(self, N=1, rng=None):
        return sps.cauchy.rvs(loc=self.location, scale=self.scale, size=(N,self.dim), random_state=rng).T
    

