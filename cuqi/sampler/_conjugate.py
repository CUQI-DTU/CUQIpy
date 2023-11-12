from cuqi.distribution import Posterior, Gaussian, Gamma, GMRF, ImplicitRegularizedGaussian, ImplicitRegularizedGMRF
import numpy as np

class Conjugate: # TODO: Subclass from Sampler once updated
    """ Conjugate sampler

    Sampler for sampling a posterior distribution where the likelihood and prior are conjugate.

    Currently supported conjugate pairs are:
    - (Gaussian, Gamma)
    - (GMRF, Gamma)
    - (ImplicitRegularizedGaussian, Gamma) with nonnegativity constraints only

    For more information on conjugate pairs, see https://en.wikipedia.org/wiki/Conjugate_prior.

    """

    def __init__(self, target: Posterior):
        if not isinstance(target.likelihood.distribution, (Gaussian, GMRF, ImplicitRegularizedGaussian, ImplicitRegularizedGMRF)):
            raise ValueError("Conjugate sampler only works with a Gaussian-type likelihood function")
        if not isinstance(target.prior, Gamma):
            raise ValueError("Conjugate sampler only works with Gamma prior")
            
        if isinstance(target.likelihood.distribution, (ImplicitRegularizedGaussian, ImplicitRegularizedGMRF)) and target.likelihood.distribution.preset not in ["nonnegativity"]:
               raise ValueError("Conjugate sampler only works implicit regularized Gaussian likelihood with nonnegativity constraints")
        
        self.target = target

    def step(self, x=None):
        if isinstance(self.target.likelihood.distribution, (Gaussian, GMRF)):
            # Extract variables
            b = self.target.likelihood.data                                 #mu
            m = len(b)                                                      #n
            Ax = self.target.likelihood.distribution.mean                   #x_i
            L = self.target.likelihood.distribution(np.array([1])).sqrtprec #L
            alpha = self.target.prior.shape                                 #alpha
            beta = self.target.prior.rate                                   #beta
    
            # Create Gamma distribution and sample
            dist = Gamma(shape=m/2+alpha,rate=.5*np.linalg.norm(L@(Ax-b))**2+beta)
    
            return dist.sample()
        elif isinstance(self.target.likelihood.distribution, (ImplicitRegularizedGaussian, ImplicitRegularizedGMRF)):
            # Extract variables
            b = self.target.likelihood.data                                          #mu
            m = len(b)                                                               #n
            Ax = self.target.likelihood.distribution.gaussian.mean                   #x_i
            L = self.target.likelihood.distribution(np.array([1])).gaussian.sqrtprec #L
            alpha = self.target.prior.shape                                          #alpha
            beta = self.target.prior.rate                                            #beta
    
            # Create Gamma distribution and sample
            dist = Gamma(shape=np.count_nonzero(b)/2+alpha,rate=.5*np.linalg.norm(L@(Ax-b))**2+beta)
            return dist.sample()
