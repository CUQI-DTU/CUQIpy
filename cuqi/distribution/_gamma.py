import numpy as np
import scipy.stats as sps
from cuqi.distribution import Distribution
from cuqi.utilities import force_ndarray

class Gamma(Distribution):
    """
    Represents a multivariate Gamma distribution characterized by shape and rate parameters of independent random variables x_i. Each is distributed according to the PDF function

    .. math::
    
        f(x_i; \\alpha, \\beta) = \\beta^\\alpha x_i^{\\alpha-1} \\exp(-\\beta x_i) / \Gamma(\\alpha)

    where shape :math:`\\alpha` and rate :math:`\\beta` are the parameters of the distribution, and :math:`\Gamma` is the Gamma function.

    In case shape and/or rate are arrays, the pdf looks like

    .. math::
    
        f(x_i; \\alpha_i, \\beta_i) = \\beta_i^{\\alpha_i} x_i^{\\alpha_i-1} \\exp(-\\beta_i x_i) / \Gamma(\\alpha_i)

    Parameters
    ----------
    shape : float or array_like, optional
        The shape parameter of the Gamma distribution. Must be positive.

    rate : float or array_like, optional
        The rate parameter of the Gamma distribution. Must be positive.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import cuqi
        import matplotlib.pyplot as plt

        # Create a multivariate Gamma distribution with the same shape and rate parameters
        shape = 1
        rate = 1e-4
        gamma_dist = cuqi.distribution.Gamma(shape=shape, rate=rate, geometry=10)

        # Generate samples
        samples = gamma_dist.sample(10000)

        # Plot histogram of samples for index 0
        samples.hist_chain(0, bins=70)


    .. code-block:: python

        import numpy as np
        import cuqi
        import matplotlib.pyplot as plt

        # Create a multivariate Gamma distribution with different shape and rate parameters
        shape = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        rate = [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5]
        gamma_dist = cuqi.distribution.Gamma(shape=shape, rate=rate)

        # Generate samples
        samples = gamma_dist.sample(10000)

        # Plot histogram of samples for index 0
        samples.hist_chain(0, bins=70)

    """
    def __init__(self, shape=None, rate=None, is_symmetric=False, **kwargs):
        # Init from abstract distribution class
        super().__init__(is_symmetric=is_symmetric,**kwargs) 

        self.shape = shape
        self.rate = rate

    @property
    def shape(self):
        """ Shape parameter of the Gamma distribution. Must be positive. """
        return self._shape
    
    @shape.setter
    def shape(self, value):
        self._shape = force_ndarray(value, flatten=True)

    @property
    def rate(self):
        """ Rate parameter of the Gamma distribution. Must be positive. """
        return self._rate
    
    @rate.setter
    def rate(self, value):
        self._rate = force_ndarray(value, flatten=True)

    @property
    def scale(self):
        """ Scale parameter of the Gamma distribution. Must be positive. This is the inverse of the rate parameter. """
        return 1/self.rate

    def logpdf(self, x):
        return np.sum(sps.gamma.logpdf(x, a=self.shape, loc=0, scale=self.scale))

    def cdf(self, x):
        return np.prod(sps.gamma.cdf(x, a=self.shape, loc=0, scale=self.scale))

    def _sample(self, N, rng=None):
        if rng is not None:
            return rng.gamma(shape=self.shape, scale=self.scale, size=(N, self.dim)).T
        else:
            return np.random.gamma(shape=self.shape, scale=self.scale, size=(N, self.dim)).T
            