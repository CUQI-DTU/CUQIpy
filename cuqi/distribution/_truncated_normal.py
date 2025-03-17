import numpy as np
from scipy.special import erf
from cuqi.utilities import force_ndarray
from cuqi.distribution import Distribution
from cuqi.distribution import Normal

class TruncatedNormal(Distribution):
    """
    Truncated Normal probability distribution.
    
    Generates instance of cuqi.distribution.TruncatedNormal. 
    It allows the user to specify upper and lower bounds on random variables 
    represented by a Normal distribution. This distribution is suitable for a 
    small dimension setup (e.g. `dim`=3 or 4). Using TruncatedNormal 
    Distribution with a larger dimension can lead to a high rejection rate when
    used within MCMC samplers.
    
    The variables of this distribution are iid.

    
    Parameters
    ------------
    mean : float or array_like of floats
        mean of distribution
    std : float or array_like of floats
        standard deviation
    low : float or array_like of floats
        lower bound of the distribution
    high : float or array_like of floats
        upper bound of the distribution
    
    Example
    -----------
    .. code-block:: python

        #Generate Normal with mean 0, standard deviation 1 and bounds [-2,2]
        p = cuqi.distribution.TruncatedNormal(mean=0, std=1, low=-2, high=2)
        samples = p.sample(5000)
    """
    def __init__(self, mean=None, std=None, low=-np.inf, high=np.inf, is_symmetric=False, **kwargs):
        # Init from abstract distribution class
        super().__init__(is_symmetric=is_symmetric, **kwargs)  

        # Init specific to this distribution
        self.mean = mean
        self.std = std
        self.low = low
        self.high = high

        # Init underlying normal distribution
        self._normal = Normal(self.mean, self.std, is_symmetric=True, **kwargs)

    @property
    def mean(self):
        """ Mean of the distribution """
        return self._mean

    @mean.setter
    def mean(self, value):
        self._mean = force_ndarray(value, flatten=True)
        if hasattr(self, '_normal'):
            self._normal.mean = self._mean

    @property
    def std(self):
        """ Std of the distribution """
        return self._std

    @std.setter
    def std(self, value):
        self._std = force_ndarray(value, flatten=True)
        if hasattr(self, '_normal'):
            self._normal.std = self._std

    @property
    def low(self):
        """ Lower bound of the distribution """
        return self._low

    @low.setter
    def low(self, value):
        self._low = force_ndarray(value, flatten=True)

    @property
    def high(self):
        """ Higher bound of the distribution """
        return self._high

    @high.setter
    def high(self, value):
        self._high = force_ndarray(value, flatten=True)

    def logpdf(self, x):
        """
        Computes the unnormalized logpdf at the given values of x.
        """
        # the unnormalized logpdf
        # check if x falls in the range between np.array a and b
        if np.any(x < self.low) or np.any(x > self.high):
            return -np.inf
        else:
             return self._normal.logpdf(x)

    def _gradient(self, x, *args, **kwargs):
        """
        Computes the gradient of the unnormalized logpdf at the given values of x.
        """
        # check if x falls in the range between np.array a and b
        if np.any(x < self.low) or np.any(x > self.high):
            return np.NaN*np.ones_like(x)
        else:
            return self._normal.gradient(x, *args, **kwargs)

    def _sample(self, N=1, rng=None):
        """
        Generates random samples from the distribution.
        """
        max_iter = 1e9 # maximum number of trials to avoid infinite loop
        samples = []
        for i in range(int(max_iter)):
            if len(samples) == N:
                break
            sample = self._normal.sample(1,rng)
            if np.all(sample >= self.low) and np.all(sample <= self.high):
                samples.append(sample)
        # raise a error if the number of iterations exceeds max_iter
        if i == max_iter-1:
            raise RuntimeError("Failed to generate {} samples within {} iterations".format(N, max_iter))
        return np.array(samples).T.reshape(-1,N)