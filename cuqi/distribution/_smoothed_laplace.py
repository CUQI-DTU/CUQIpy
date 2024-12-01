import numpy as np
from cuqi.utilities import force_ndarray
from cuqi.distribution import Distribution

class SmoothedLaplace(Distribution):
    """ Smoothed Laplace distribution. 

    Defines a smoothed Laplace distribution given a location, a scale and a smoothing parameter
    beta. The smoothed Laplace distribution is defined as

    .. math::

        p(x) = \\frac{1}{2b} \exp\left(-\\frac{\sqrt{(x-\mu)^2 + \\beta}}{b}\\right),

    where :math:`\mu` is the location (mean), :math:`b` is the scale (decay) parameter and
    :math:`\\beta` is the smoothing parameter.

    The rate parameter is defined as :math:`\lambda = \\frac{1}{b}`.

    The variables of this Laplace distribution are independent identically distributed (i.i.d.).

    Parameters
    ----------
    location : scalar, list, tuple, or ndarray
        The location parameter of the distribution.

    scale : scalar, list, tuple, or ndarray
        The scale parameter of the distribution.
    
    beta : scalar
        The smoothing parameter of the distribution.

    """

    def __init__(self, location=None, scale=None, beta=1e-3, **kwargs):
        super().__init__(**kwargs)

        self.location = location
        self.scale = scale
        self.beta = beta

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

    @property
    def beta(self):
        """ Beta parameter """
        return self._beta
    
    @beta.setter
    def beta(self, value):
        self._beta = value

    def logpdf(self, x):
        """
        Computes the logarithm of the probability density function at the given values of x.
        """
        # x accepts scalar, list, tuple, or ndarray
        if isinstance(x, (float, int)):
            x = np.array([x])
        elif isinstance(x, (list, tuple)):
            x = np.array(x)
        return np.sum(np.log(0.5 / self.scale)) - np.sum(np.sqrt((x - self.location) ** 2 + self.beta) / self.scale)

    def gradient(self, x):
        """
        Computes the gradient of logpdf at the given values of x.
        """
        # x accepts scalar, list, tuple, or ndarray
        if isinstance(x, (float, int)):
            x = np.array([x])
        elif isinstance(x, (list, tuple)):
            x = np.array(x)
        return -np.array((x - self.location) / self.scale / np.sqrt((x - self.location) ** 2 + self.beta))

    def _sample(self, N=1, rng=None):
        """
        Generates random samples from the distribution.
        """
        raise NotImplementedError(f"sample is not implemented for {self.__class__.__name__}.")
