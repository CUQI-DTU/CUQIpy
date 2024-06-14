import numpy as np
from cuqi.experimental.mcmc import SamplerNew
from cuqi.distribution import Posterior, LMRF, Gamma
import scipy as sp

class ConjugateApproxNew(SamplerNew):
    """ Approximate Conjugate sampler

    Sampler for sampling a posterior distribution where the likelihood and prior can be approximated
    by a conjugate pair.

    Currently supported pairs are:
    - (LMRF, Gamma): Approximated by (Gaussian, Gamma) where Gamma is defined on the inverse of the scale parameter of the LMRF distribution.

    Gamma distribution must be univariate.

    LMRF likelihood must have zero mean.

    Currently, the sampler does NOT automatically check that the conjugate distributions are defined on the correct parameters.


    For more information on conjugate pairs, see https://en.wikipedia.org/wiki/Conjugate_prior.

    """

    def _initialize(self):
        pass

    def validate_target(self):

        if not isinstance(self.target, Posterior):
            raise TypeError("Approximate conjugate sampler requires a target of type Posterior")

        if not isinstance(self.target.likelihood.distribution, LMRF):
            raise ValueError("Approximate conjugate sampler only works with LMRF likelihood function")
        
        if not isinstance(self.target.prior, Gamma):
            raise ValueError("Approximate conjugate sampler only works with Gamma prior")
        
        if not self.target.prior.dim == 1:
            raise ValueError("Approximate conjugate sampler only works with univariate Gamma prior")
        
        if np.sum(self.target.likelihood.distribution.location) != 0:
            raise ValueError("Approximate conjugate sampler only works with zero mean LMRF likelihood")
        
    def step(self):
        # Extract variables
        # Here we approximate the LMRF with a Gaussian

        # Extract diff_op from target likelihood
        D = self.target.likelihood.distribution._diff_op
        n = D.shape[0]

        # Gaussian approximation of LMRF prior as function of x_k
        # See Uribe et al. (2022) for details
        # Current has a zero mean assumption on likelihood! TODO
        beta=1e-5
        def Lk_fun(x_k):
            dd =  1/np.sqrt((D @ x_k)**2 + beta*np.ones(n))
            W = sp.sparse.diags(dd)
            return W.sqrt() @ D

        x = self.target.likelihood.data                 #x
        d = len(x)                                      #d
        Lx = Lk_fun(x)@x                                #Lx
        alpha = self.target.prior.shape                 #alpha
        beta = self.target.prior.rate                   #beta

        # Create Gamma distribution and sample
        dist = Gamma(shape=d+alpha, rate=np.linalg.norm(Lx)**2+beta)

        self.current_point = dist.sample()

    def tune(self, skip_len, update_count):
        pass
