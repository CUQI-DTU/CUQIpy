from abc import ABC, abstractmethod
import sys

from cuqi.sampler import Sampler
from cuqi.distribution import Posterior, Gaussian, Gamma, GMRF
from cuqi.implicitprior import RegularizedGaussian, RegularizedGMRF
from cuqi.utilities import get_non_default_args

import numpy as np
from math import isclose

class Conjugate: # TODO: Subclass from Sampler once updated
    """ Conjugate sampler

    Sampler for sampling a posterior distribution where the likelihood and prior are conjugate.

    Currently supported conjugate pairs are:
    - (Gaussian, Gamma)
    - (GMRF, Gamma)
    - (RegularizedGaussian, Gamma) with nonnegativity constraints only
    - (RegularizedGMRF, Gamma) with nonnegativity constraints only

    For more information on conjugate pairs, see https://en.wikipedia.org/wiki/Conjugate_prior.

    For implicit regularized Gaussians see:
    
    [1] Everink, Jasper M., Yiqiu Dong, and Martin S. Andersen. "Bayesian inference with projected densities." SIAM/ASA Journal on Uncertainty Quantification 11.3 (2023): 1025-1043.

    """

    def __init__(self, target: Posterior):

        if isinstance(target.likelihood.distribution, (Gaussian, GMRF)) and isinstance(target.prior, (Gamma)):
            self.conjugacypair = GaussianGammaPair(target)
        elif isinstance(target.likelihood.distribution, (RegularizedGaussian, RegularizedGMRF)) and isinstance(target.prior, (Gamma)):
            self.conjugacypair = RegularizedGaussianGammaPair(target)
        elif isinstance(target.likelihood.distribution, (Gamma)) and isinstance(target.prior, (Gamma)):
            self.conjugacypair = GammaGammaPair(target)
        else:
            raise ValueError(f"Conjugate does not support a conjugacy pair with likelihood {type(target.likelihood.distribution)} and prior {type(target.prior)}")
        
        self.target = target


    def step(self, x=None):
        return self.conjugacypair.step(x)
    
    def validate(self):
        return self.conjugacypair.validate()

class ConjugatePair(ABC):

    def __init__(self, target: Posterior):
        self.target = target
        self.conjugate_variable = target.prior.name

    @property
    @abstractmethod
    def distribution(self):
        pass

    def step(self, x=None):
        return self.distribution.sample()

    @abstractmethod
    def _validate(self, name, attr):
        return True

    def validate(self):
        mutable_likelihood_vars = self.target.likelihood.distribution.get_mutable_variables()
        
        # Check only those attributes that are functions and have the conjugacy variable as argument
        for var_key in mutable_likelihood_vars:
            attr = getattr(self.target.likelihood.distribution, var_key)
            if callable(attr) and self.conjugate_variable in get_non_default_args(attr):
                if not self._validate(var_key, attr):
                    raise ValueError(f"Conjugacy on the variable {self.conjugate_variable} in attribute {var_key} is not incorrect or unsupported")
               
        return True

# Tests whether a function (scalar to scalar) is the idenity (lambda x : x)
def regression_test_scalar_identity(f):
    return all([f(1.0) == 1.0,
                f(10.0) == 10.0,
                f(100.0) == 100.0])

def regression_test_linear(f):
    reference_f = f(1.0)
    if isinstance(reference_f,np.ndarray):
        reference = f(1.0).flatten()[0]
        return all([f(10.0).flatten()[0] == 10.0*reference,
                    f(100.0).flatten()[0] == 100.0*reference])
    else:
        reference = f(1.0)
        return all([f(10.0) == 10.0*reference,
                    f(100.0) == 100.0*reference])


# Tests whether a function (scalar to scalar) is the reciprocal (lambda x : 1.0/x)
def regression_test_scalar_reciprocal(f):
    return all([isclose(f(1.0), 1.0),
                isclose(f(10.0), 1.0/10.0),
                isclose(f(100.0), 1.0/100.0)])


class GaussianGammaPair(ConjugatePair):

    def __init__(self, target: Posterior):
        if not isinstance(target.likelihood.distribution, (Gaussian, GMRF)):
            raise ValueError(f"The likelihood needs to be Gaussian or GMRF, but is {type(target.likelihood.distribution)}")
        if not isinstance(target.prior, (Gamma)):
            raise ValueError(f"The prior needs to be Gamma , but is {type(target.prior)}")
        
        super().__init__(target)

    @property
    def distribution(self):
        # Extract variables
        b = self.target.likelihood.data                                 #mu
        m = len(b)                                                      #n
        Ax = self.target.likelihood.distribution.mean                   #x_i
        L = self.target.likelihood.distribution(np.array([1])).sqrtprec #L
        alpha = self.target.prior.shape                                 #alpha
        beta = self.target.prior.rate                                   #beta

        # Create Gamma distribution 
        return Gamma(shape=m/2+alpha,rate=.5*np.linalg.norm(L@(Ax-b))**2+beta)
    
    def _validate(self, name, attr):
        if name in ["cov"]:
            return regression_test_scalar_reciprocal(attr)
        if name in ["prec"]:
            return regression_test_linear(attr)
        raise ValueError(f"Conjugate variable {self.conjugate_variable} in attribute {name} is unsupported.")
    
    
class RegularizedGaussianGammaPair(ConjugatePair):

    def __init__(self, target: Posterior):
        if not isinstance(target.likelihood.distribution, (RegularizedGaussian, RegularizedGMRF)):
            raise ValueError(f"The likelihood needs to be RegularizedGaussian or RegularizedGMRF, but is {type(target.likelihood.distribution)}")
        if not isinstance(target.prior, (Gamma)):
            raise ValueError(f"The prior needs to be Gamma, but is {type(target.prior)}")

        super().__init__(target)

    @property
    def distribution(self):
        # Extract variables
        b = self.target.likelihood.data                                 #mu
        m = np.count_nonzero(b)                                         #n
        Ax = self.target.likelihood.distribution.mean                   #x_i
        L = self.target.likelihood.distribution(np.array([1])).sqrtprec #L
        alpha = self.target.prior.shape                                 #alpha
        beta = self.target.prior.rate                                   #beta

        # Create Gamma distribution
        return Gamma(shape=m/2+alpha,rate=.5*np.linalg.norm(L@(Ax-b))**2+beta)

    
    def _validate(self, name, attr):
        if name in ["cov"]:
            return regression_test_scalar_reciprocal(attr)
        if name in ["prec"]:
            return regression_test_linear(attr)
        raise ValueError(f"Conjugate variable {self.conjugate_variable} in attribute {name} is unsupported.")
    

class GammaGammaPair(ConjugatePair):

    def __init__(self, target: Posterior):
        if not isinstance(target.likelihood.distribution, (Gamma)):
            raise ValueError(f"The likelihood needs to be Gamma, but is {type(target.likelihood.distribution)}")
        if not isinstance(target.prior, (Gamma)):
            raise ValueError(f"The prior needs to be Gamma, but is {type(target.prior)}")

        super().__init__(target)

    @property
    def distribution(self):
        # Extract variables
        shape_p = self.target.prior.shape
        rate_p = self.target.prior.rate

        likelihood = self.target.likelihood.distribution(np.ones_like(rate_p))
        shape_l = likelihood.shape
        rate_l = likelihood.rate

        # Create Gamma distribution
        return Gamma(shape=shape_l + shape_p, rate=rate_l*self.target.likelihood.data + rate_p)
    
    def _validate(self, name, attr):
        if name in ["rate"]:
            return regression_test_linear(attr)
        raise ValueError(f"Conjugate variable {self.conjugate_variable} in attribute {name} is unsupported.")