import numpy as np
from cuqi.geometry import _get_identity_geometries
from cuqi.distribution import Distribution
from cuqi.distribution import Gaussian
import warnings

class Lognormal(Distribution):
    """
    Multivariate Lognormal distribution

    Parameters
    ------------
    mean: np.ndarray
        Mean of the normal distribution used to define the lognormal distribution 

    cov: np.ndarray
        Covariance matrix of the normal distribution used to define the lognormal distribution 
    
    Example
    -------
    .. code-block:: python
    
        # Generate a lognormal distribution
        mean = np.array([1.5,1])
        cov = np.array([[3, 0],[0, 1]])
        x = cuqi.distribution.Lognormal(mean, cov)
        samples = x.sample(10000)
        samples.hist_chain(1, bins=70)

    """
    def __init__(self, mean, cov, is_symmetric=False, **kwargs):
        super().__init__(is_symmetric=is_symmetric, **kwargs) 
        self.mean = mean
        self.cov = cov
        self._normal = Gaussian(self.mean, self.cov)

    @property
    def _normal(self):
        if not np.all(self._Gaussian.mean == self.mean):
            self._Gaussian.mean = self.mean
        if not np.all(self._Gaussian.cov == self.cov):
            self._Gaussian.cov = self.cov 
        return self._Gaussian

    @_normal.setter
    def _normal(self, value):
        self._Gaussian = value

    @property
    def dim(self):
        return self._normal.dim

    def pdf(self, x):
        if np.any(x<=0):
            return 0
        else:
            return self._normal.pdf(np.log(x))*np.prod(1/x)

    def logpdf(self, x):
        return np.log(self.pdf(x))

    def _gradient(self, val, *args, **kwargs):
        #Avoid complicated geometries that change the gradient.
        if not type(self.geometry) in _get_identity_geometries():
            raise NotImplementedError("Gradient not implemented for distribution {} "
                                      "with geometry {}".format(self,self.geometry))

        elif not callable(self._normal.mean): # for prior
            return np.diag(1/val)@(-1+self._normal.gradient(np.log(val)))
        elif hasattr(self.mean,"gradient"): # for likelihood
            model = self._normal.mean
            dev = np.log(val) - model.forward(*args, **kwargs)
            return  model.gradient(self._normal.prec@dev, *args, **kwargs) # Jac(x).T@(self._normal.prec@dev)
        else:
            warnings.warn('Gradient not implemented for {}'.format(type(self._normal.mean)))

    def _sample(self, N=1, rng=None):
        return np.exp(self._normal._sample(N,rng))
