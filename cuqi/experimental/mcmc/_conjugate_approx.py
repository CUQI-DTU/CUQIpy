import numpy as np
from cuqi.experimental.mcmc import Conjugate
from cuqi.experimental.mcmc._conjugate import _ConjugatePair, _get_conjugate_parameter, _check_conjugate_parameter_is_scalar_reciprocal
from cuqi.distribution import LMRF, Gamma
import scipy as sp

class ConjugateApprox(Conjugate):
    """ Approximate Conjugate sampler

    Sampler for sampling a posterior distribution where the likelihood and prior can be approximated
    by a conjugate pair.

    Currently supported pairs are:
    - (LMRF, Gamma): Approximated by (Gaussian, Gamma) where Gamma is defined on the inverse of the scale parameter of the LMRF distribution.

    Gamma distribution must be univariate.

    LMRF likelihood must have zero mean.

    For more details on conjugacy see :class:`Conjugate`.

    """

    def _set_conjugatepair(self):
        """ Set the conjugate pair based on the likelihood and prior. This requires target to be set. """
        if isinstance(self.target.likelihood.distribution, LMRF) and isinstance(self.target.prior, Gamma):
            self._conjugatepair = _LMRFGammaPair(self.target)
        else:
            raise ValueError(f"Conjugacy is not defined for likelihood {type(self.target.likelihood.distribution)} and prior {type(self.target.prior)}, in CUQIpy")


class _LMRFGammaPair(_ConjugatePair):
    """ Implementation of the conjugate pair (LMRF, Gamma) """

    def validate_target(self):
        if not self.target.prior.dim == 1:
            raise ValueError("Approximate conjugate sampler only works with univariate Gamma prior")
        
        if np.sum(self.target.likelihood.distribution.location) != 0:
            raise ValueError("Approximate conjugate sampler only works with zero mean LMRF likelihood")
        
        key_value_pairs = _get_conjugate_parameter(self.target)
        if len(key_value_pairs) != 1:
            raise ValueError(f"Multiple references to conjugate parameter {self.target.prior.name} found in likelihood. Only one occurance is supported.")
        for key, value in key_value_pairs:
            if key == "scale":
                if not _check_conjugate_parameter_is_scalar_reciprocal(value):
                    raise ValueError("Approximate conjugate sampler only works with Gamma prior on the inverse of the scale parameter of the LMRF likelihood")
            else:
                raise ValueError(f"No approximate conjugacy defined for likelihood {type(self.target.likelihood.distribution)} and prior {type(self.target.prior)}, in CUQIpy")
        
    def conjugate_distribution(self):
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
        return Gamma(shape=d+alpha, rate=np.linalg.norm(Lx)**2+beta)
