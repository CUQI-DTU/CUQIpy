from cuqi.distribution import Posterior, GaussianCov, GaussianPrec, Gamma, GMRF
import numpy as np

class Conjugate:
    """ Conjugate sampler
    
    https://en.wikipedia.org/wiki/Conjugate_prior
    GaussianPrec likelihood + Gamma prior yields Gamma( alpha+n/2, beta+sum((x_i-mu)^2)/2) )
    """

    
    def __init__(self, target: Posterior):
        if not isinstance(target.likelihood.distribution, (GaussianPrec, GaussianCov, GMRF)):
            raise ValueError("Conjugate sampler only works with GaussianPrec likelihood function")
        if not isinstance(target.prior, Gamma):
            raise ValueError("Conjugate sampler only works with Gamma prior")
        self.target = target

    def step(self, x=None):
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
