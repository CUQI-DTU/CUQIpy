import numpy as np
from cuqi.experimental.mcmc import SamplerNew
from cuqi.distribution import Posterior, Gaussian, Gamma, GMRF
from cuqi.implicitprior import RegularizedGaussian, RegularizedGMRF

class ConjugateNew(SamplerNew):

    def __init__(self, target:Posterior=None):
        self.target = target

    def validate_target(self):

        if not isinstance(self.target.likelihood.distribution, (Gaussian, GMRF, RegularizedGaussian, RegularizedGMRF)):
            raise ValueError("Conjugate sampler only works with a Gaussian-type likelihood function")
        
        if not isinstance(self.target.prior, Gamma):
            raise ValueError("Conjugate sampler only works with Gamma prior")
        
        if not self.target.prior.dim == 1:
            raise ValueError("Conjugate sampler only works with univariate Gamma prior")
        
        if isinstance(self.target.likelihood.distribution, (RegularizedGaussian, RegularizedGMRF)) and self.target.likelihood.distribution.preset not in ["nonnegativity"]:
            raise ValueError("Conjugate sampler only works implicit regularized Gaussian likelihood with nonnegativity constraints")

    def step(self):
        # Extract variables
        b = self.target.likelihood.data                                 #mu
        m = self._calc_m_for_Gaussians(b)                               #n
        Ax = self.target.likelihood.distribution.mean                   #x_i
        L = self.target.likelihood.distribution(np.array([1])).sqrtprec #L
        alpha = self.target.prior.shape                                 #alpha
        beta = self.target.prior.rate                                   #beta

        # Create Gamma distribution and sample
        dist = Gamma(shape=m/2+alpha,rate=.5*np.linalg.norm(L@(Ax-b))**2+beta)

        self.current_point = dist.sample()

    def tune(self, skip_len, update_count):
        pass

    def _calc_m_for_Gaussians(self, b):
        """ Helper method to calculate m parameter for Gaussian-Gamma conjugate pair. """
        if isinstance(self.target.likelihood.distribution, (Gaussian, GMRF)):
            return len(b)
        elif isinstance(self.target.likelihood.distribution, (RegularizedGaussian, RegularizedGMRF)):
            return np.count_nonzero(b) # See 
