import numpy as np
import scipy.stats as sps
from scipy.special import loggamma, gammainc
from cuqi.distribution import Distribution

class Gamma(Distribution):
    """
    Represents a multivariate Gamma distribution characterized by shape and rate parameters of independent random variables x_i. Each is distributed according to the PDF function
    
    f(x; shape, rate) = rate^shape * x^(shape-1) * exp(-rate * x) / Gamma(shape)

    where `shape` and `rate` are the parameters of the distribution, and Gamma is the Gamma function.

    Parameters
    ----------
    shape : float or array_like, optional
        The shape parameter of the Gamma distribution. Must be positive.

    rate : float or array_like, optional
        The rate parameter of the Gamma distribution. Must be positive. The inverse of the scale parameter.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import cuqi
        import matplotlib.pyplot as plt

        # Create a Gamma distribution instance
        shape = 1
        rate = 1e-4
        gamma_dist = cuqi.distribution.Gamma(shape=shape, rate=rate)

        # Generate samples
        samples = gamma_dist.sample(10000)

        # Plot histogram of samples
        samples.hist_chain(0, bins=70)
    """
    def __init__(self, shape=None, rate=None, is_symmetric=False, **kwargs):
        # Init from abstract distribution class
        super().__init__(is_symmetric=is_symmetric,**kwargs) 

        # Init specific to this distribution
        self.shape = shape
        self.rate = rate     

    @property
    def scale(self):
        return 1/self.rate

    def logpdf(self, x):
        #return (self.shape*np.log(self.rate)-loggamma(self.shape)) + ((self.shape-1)*np.log(x) - self.rate*x)
        return np.sum(sps.gamma.logpdf(x, a=self.shape, loc=0, scale=self.scale))

    def cdf(self, x):
        #return gammainc(self.shape, self.rate*x)
        return np.prod(sps.gamma.cdf(x, a=self.shape, loc=0, scale=self.scale))

    def _sample(self, N, rng=None):
        if rng is not None:
            return rng.gamma(shape=self.shape, scale=self.scale, size=(N, self.dim)).T
        else:
            return np.random.gamma(shape=self.shape, scale=self.scale, size=(N, self.dim)).T
            