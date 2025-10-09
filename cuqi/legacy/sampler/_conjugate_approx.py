from cuqi.distribution import Posterior, LMRF, Gamma
import numpy as np
import scipy as sp

class ConjugateApprox: # TODO: Subclass from Sampler once updated
    """ Approximate Conjugate sampler

    Sampler for sampling a posterior distribution where the likelihood and prior can be approximated
    by a conjugate pair.

    Currently supported pairs are:
    - (LMRF, Gamma): Approximated by (Gaussian, Gamma)

    For more information on conjugate pairs, see https://en.wikipedia.org/wiki/Conjugate_prior.

    """

    
    def __init__(self, target: Posterior):
        if not isinstance(target.likelihood.distribution, LMRF):
            raise ValueError("Conjugate sampler only works with Laplace diff likelihood function")
        if not isinstance(target.prior, Gamma):
            raise ValueError("Conjugate sampler only works with Gamma prior")
        self.target = target

    def step(self, x=None):
        # Extract variables
        # Here we approximate the Laplace diff with a Gaussian

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

        return dist.sample()
