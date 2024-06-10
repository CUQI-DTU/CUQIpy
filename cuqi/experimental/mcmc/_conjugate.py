import numpy as np
from cuqi.experimental.mcmc import SamplerNew
from cuqi.distribution import Posterior, Gaussian, Gamma, GMRF
from cuqi.implicitprior import RegularizedGaussian, RegularizedGMRF

class ConjugateNew(SamplerNew):
    """ Conjugate sampler

    Sampler for sampling a posterior distribution where the likelihood and prior are conjugate.

    Currently supported conjugate pairs are:
    - (Gaussian, Gamma) where Gamma is defined on the precision parameter of the Gaussian
    - (GMRF, Gamma) where Gamma is defined on the precision parameter of the GMRF
    - (RegularizedGaussian, Gamma) with nonnegativity constraints only and Gamma is defined on the precision parameter of the RegularizedGaussian
    - (RegularizedGMRF, Gamma) with nonnegativity constraints only and Gamma is defined on the precision parameter of the RegularizedGMRF

    Gamma distribution must be univariate.

    Currently, the sampler does NOT automatically check that the conjugate distributions are defined on the correct parameters.

    For more information on conjugate pairs, see https://en.wikipedia.org/wiki/Conjugate_prior.

    For implicit regularized Gaussians see:
    
    [1] Everink, Jasper M., Yiqiu Dong, and Martin S. Andersen. "Bayesian inference with projected densities." SIAM/ASA Journal on Uncertainty Quantification 11.3 (2023): 1025-1043.

    """
    def _initialize(self):
        pass

    def validate_target(self):

        if not isinstance(self.target, Posterior):
            raise TypeError("Conjugate sampler requires a target of type Posterior")

        if not isinstance(self.target.likelihood.distribution, (Gaussian, GMRF, RegularizedGaussian, RegularizedGMRF)):
            raise ValueError("Conjugate sampler only works with a Gaussian-type likelihood function")
        
        if not isinstance(self.target.prior, Gamma):
            raise ValueError("Conjugate sampler only works with Gamma prior")
        
        if not self.target.prior.dim == 1:
            raise ValueError("Conjugate sampler only works with univariate Gamma prior")
        
        if isinstance(self.target.likelihood.distribution, (RegularizedGaussian, RegularizedGMRF)) and self.target.likelihood.distribution.preset not in ["nonnegativity"]:
            raise ValueError("Conjugate sampler only works with implicit regularized Gaussian likelihood with nonnegativity constraints")

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
        """ Helper method to calculate m parameter for Gaussian-Gamma conjugate pair.
         
        Classically m defines the number of observations in the Gaussian likelihood function.

        However, for implicit regularized Gaussians, m is the number of non-zero elements in the data vector b see [1].
           
        """

        if isinstance(self.target.likelihood.distribution, (Gaussian, GMRF)):
            return len(b)
        elif isinstance(self.target.likelihood.distribution, (RegularizedGaussian, RegularizedGMRF)):
            return np.count_nonzero(b)
