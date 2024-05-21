from cuqi.distribution import Posterior, Gaussian, Gamma, GMRF
from cuqi.implicitprior import RegularizedGaussian, RegularizedGMRF, RegularizedUniform
import numpy as np

class Conjugate: # TODO: Subclass from Sampler once updated
    """ Conjugate sampler

    Sampler for sampling a posterior distribution where the likelihood and prior are conjugate.

    Currently supported conjugate pairs are:
    - (Gaussian, Gamma)
    - (GMRF, Gamma)
    - (RegularizedGaussian, Gamma) with nonnegativity constraints only
    - (RegularizedGMRF, Gamma) with nonnegativity constraints only
    - (RegularizedUniform, Gamma) with l1 regularization only

    For more information on conjugate pairs, see https://en.wikipedia.org/wiki/Conjugate_prior.

    For implicit regularized Gaussians see:
    
    [1] Everink, Jasper M., Yiqiu Dong, and Martin S. Andersen. "Bayesian inference with projected densities." SIAM/ASA Journal on Uncertainty Quantification 11.3 (2023): 1025-1043.

    """

    def __init__(self, target: Posterior):
        if not isinstance(target.likelihood.distribution, (Gaussian, GMRF, RegularizedGaussian, RegularizedGMRF, RegularizedUniform)):
            raise ValueError("Conjugate sampler only works with a Gaussian-type likelihood function")
        if not isinstance(target.prior, Gamma):
            raise ValueError("Conjugate sampler only works with Gamma prior")
        if not target.prior.dim == 1:
            raise ValueError("Conjugate sampler only works with univariate Gamma prior")
            
        if isinstance(target.likelihood.distribution, (RegularizedGaussian, RegularizedGMRF)) and not isinstance(target.likelihood.distribution, (RegularizedUniform)) and target.likelihood.distribution.preset not in ["nonnegativity", "l1", "TV"]:
            raise ValueError("Conjugate sampler only works implicit regularized Gaussian likelihood with nonnegativity constraints")
        if isinstance(target.likelihood.distribution, (RegularizedUniform)) and target.likelihood.distribution.preset not in ["l1", "TV"]:
            raise ValueError("Conjugate sampler only works implicit regularized Uniform likelihood with l1 regularization")
        
        self.target = target

    def step(self, x=None):
        # Extract variables
        b = self.target.likelihood.data                                 #mu
        m = self._calc_m_for_Gaussians(b)                               #n
        Ax = self.target.likelihood.distribution.mean                   #x_i
        L = self.target.likelihood.distribution(np.array([1])).sqrtprec #L
        alpha = self.target.prior.shape                                 #alpha
        beta = self.target.prior.rate                                   #beta
        
        if isinstance(self.target.likelihood.distribution, RegularizedGaussian) and self.target.likelihood.distribution.preset in ["l1", "TV"]:
            s = self.target.likelihood.distribution(np.array([1])).strength[0]
            base_rate = s*np.linalg.norm(Ax, ord = 1) # I MADE THE *** MISTAKE AGAIN USING x!!!!!!!!!!!!!!
        else:
            base_rate = .5*np.linalg.norm(L@(Ax-b))**2

        # Create Gamma distribution and sample
        dist = Gamma(shape=m/2+alpha, rate=base_rate+beta)

        return dist.sample()

    def _calc_m_for_Gaussians(self, b):
        """ Helper method to calculate m parameter for Gaussian-Gamma conjugate pair. """
        if isinstance(self.target.likelihood.distribution, (Gaussian, GMRF)):
            return len(b)
        elif isinstance(self.target.likelihood.distribution, (RegularizedGaussian, RegularizedGMRF)):
            if self.target.likelihood.distribution.preset == "nonnegativity":
                return np.count_nonzero(b)
            if self.target.likelihood.distribution.preset in ["l1", "TV"]: # TV has a different value of m, but this is easier for testing.
                return 2*np.count_nonzero(b)
            
        raise Exception("Conjugacy pair not supported... somehow...")
