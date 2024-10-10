import numpy as np
import scipy.stats as sps
from cuqi.geometry import _get_identity_geometries
from cuqi.utilities import force_ndarray
from cuqi.distribution import Distribution

class InverseGamma(Distribution):
    """
    Multivariate inverse gamma distribution of independent random variables x_i. Each is distributed according to the PDF function

    .. math::

        f(x) = (x-\\beta)^{(-\\alpha-1)} * \exp(-\\gamma/(x-\\beta)) / (\\gamma^{(-\\alpha)}*\Gamma(\\alpha))

    where shape :math:`\\alpha`, location :math:`\\beta` and scale :math:`\\gamma` are the shape, location and scale of x_i, respectively. And :math:`\Gamma` is the Gamma function.

    Parameters
    ------------
    shape: float or array_like
        The shape parameter

    location: float or array_like
        The location of the inverse gamma distribution. The support of the pdf function is the Cartesian product of the open intervals (location_1, infinity), (location_2, infinity), ..., (location_dim, infinity).

    scale: float or array_like
        The scale of the inverse gamma distribution (non-negative)

    Example
    -------
    .. code-block:: python

        # Generate an InverseGamma distribution
        import numpy as np
        import cuqi
        import matplotlib.pyplot as plt
        shape = [1,2]
        location = 0
        scale = 1
        rng = np.random.RandomState(1)
        x = cuqi.distribution.InverseGamma(shape, location, scale)
        samples = x.sample(1000, rng=rng)
        samples.hist_chain(0, bins=70)
        plt.figure()
        samples.hist_chain(1, bins=70)

    """
    def __init__(self, shape=None, location=None, scale=None, is_symmetric=False, **kwargs):
        super().__init__(is_symmetric=is_symmetric, **kwargs) 
        self.shape = force_ndarray(shape, flatten=True)
        self.location = force_ndarray(location, flatten=True)
        self.scale = force_ndarray(scale, flatten=True)
    
    def logpdf(self, x):
        return np.sum(sps.invgamma.logpdf(x, a=self.shape, loc=self.location, scale=self.scale))

    def cdf(self, x):
        return np.prod(sps.invgamma.cdf(x, a=self.shape, loc=self.location, scale=self.scale))

    def _gradient(self, val, **kwargs):
        #Avoid complicated geometries that change the gradient.
        if not type(self.geometry) in _get_identity_geometries():
            raise NotImplementedError("Gradient not implemented for distribution {} with geometry {}".format(self,self.geometry))
        #Computing the gradient for conditional InverseGamma distribution is not supported yet    
        elif self.is_cond:
            raise NotImplementedError(f"Gradient is not implemented for {self} with conditioning variables {self.get_conditioning_variables()}")
        
        #Compute the gradient
        if np.any(val <= self.location):
            return val*np.nan
        else:
            return (-self.shape-1)/(val - self.location) +\
                    self.scale/(val - self.location)**2


    def _sample(self, N=1, rng=None):
        return sps.invgamma.rvs(a=self.shape, loc= self.location, scale = self.scale ,size=(N,self.dim), random_state=rng).T
