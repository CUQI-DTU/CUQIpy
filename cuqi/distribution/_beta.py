import numpy as np
import scipy.stats as sps
from cuqi.geometry import _get_identity_geometries
from cuqi.utilities import force_ndarray
from cuqi.distribution import Distribution


class Beta(Distribution):
    """
    Multivariate beta distribution of independent random variables x_i. Each is distributed according to the PDF function

    .. math::

        f(x) = x^{(\\alpha-1)}(1-x)^{(\\beta-1)}\Gamma(\\alpha+\\beta) / (\Gamma(\\alpha)\Gamma(\\beta))

    where :math:`\Gamma` is the Gamma function.

    Parameters
    ------------
    alpha: float or array_like
           The shape parameter :math:`\\alpha` of the beta distribution.

    beta: float or array_like
          The shape parameter :math:`\\beta` of the beta distribution.
    
    Example
    -------
    .. code-block:: python

        # % Generate a beta distribution
        import numpy as np
        import cuqi
        import matplotlib.pyplot as plt
        alpha = 0.5
        beta  = 0.5
        rng = np.random.RandomState(1)
        x = cuqi.distribution.Beta(alpha, beta)
        samples = x.sample(1000, rng=rng)
        samples.hist_chain(0, bins=70)

    """
    def __init__(self, alpha=None, beta=None, is_symmetric=False, **kwargs):
        super().__init__(is_symmetric=is_symmetric, **kwargs)
        self.alpha = force_ndarray(alpha, flatten=True)
        self.beta = force_ndarray(beta, flatten=True)

    def logpdf(self, x):

        # Check bounds
        if np.any(x<=0) or np.any(x>=1) or np.any(self.alpha<=0) or np.any(self.beta<=0):
            return -np.inf

        # Compute logpdf
        return np.sum(sps.beta.logpdf(x, a=self.alpha, b=self.beta))

    def cdf(self, x):

        # Check bounds
        if np.any(x<=0) or np.any(x>=1) or np.any(self.alpha<=0) or np.any(self.beta<=0):
            return 0

        # Compute logpdf
        return np.prod(sps.beta.cdf(x, a=self.alpha, b=self.beta))

    def _sample(self, N=1, rng=None):
        return sps.beta.rvs(a=self.alpha, b=self.beta, size=(N,self.dim), random_state=rng).T

    def _gradient(self, x, *args, **kwargs):
        #Avoid complicated geometries that change the gradient.
        if not type(self.geometry) in _get_identity_geometries():
            raise NotImplementedError("Gradient not implemented for distribution {} with geometry {}".format(self,self.geometry))
        
        #Computing the gradient for conditional InverseGamma distribution is not supported yet    
        if self.is_cond:
            raise NotImplementedError(f"Gradient is not implemented for {self} with conditioning variables {self.get_conditioning_variables()}")
        
        # Check bounds (return nan if out of bounds)
        if np.any(x<=0) or np.any(x>=1) or np.any(self.alpha<=0) or np.any(self.beta<=0):
            return x*np.nan

        #Compute the gradient
        return (self.alpha - 1)/x + (self.beta-1)/(x-1)
        