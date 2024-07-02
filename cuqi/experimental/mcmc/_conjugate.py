import sys
import inspect
import numpy as np
from abc import ABC, abstractmethod
import math
from cuqi.experimental.mcmc import SamplerNew
from cuqi.distribution import Posterior, Gaussian, Gamma, GMRF
from cuqi.implicitprior import RegularizedGaussian, RegularizedGMRF
from cuqi.utilities import get_non_default_args


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

    @SamplerNew.target.setter # Overwrite the target setter to set the conjugate pair
    def target(self, value):
        """ Set the target density. Runs validation of the target. """
        self._target = value
        if self._target is not None:
            self._set_conjugatepair()
            self.validate_target()

    def validate_target(self):
        self._ensure_target_is_posterior()
        self._conjugatepair.validate_target()
        
    def step(self):
        self.current_point = self._conjugatepair.sample()

    def tune(self, skip_len, update_count):
        pass # No tuning required for conjugate sampler

    def _ensure_target_is_posterior(self):
        """ Ensure that the target is a Posterior distribution. """
        if not isinstance(self.target, Posterior):
            raise TypeError("Conjugate sampler requires a target of type Posterior")

    def _set_conjugatepair(self):
        """ Set the conjugate pair based on the likelihood and prior. This requires target to be set. """
        self._ensure_target_is_posterior()
        for conjugate_pair in _get_all_conjugate_pairs():
            try:
                pair = conjugate_pair(self.target)
                pair.validate_target()
                self._conjugatepair = pair
                return
            except Exception: # Catch any exception since we are brute force trying all conjugate pairs
                pass
        # No conjugate pair found
        raise ValueError(f"Conjugacy is not defined for likelihood {type(self.target.likelihood.distribution)} and prior {type(self.target.prior)}, in CUQIpy")

    def __repr__(self):
        msg = super().__repr__()
        if hasattr(self, "_conjugatepair"):
            msg += f"\n Conjugate pair:\n\t {type(self._conjugatepair).__name__.removeprefix('_')}"
        return msg
    
def _get_all_conjugate_pairs():
    """Get all available conjugate pairs."""
    
    # Find all classes in the current module
    all_classes = inspect.getmembers(sys.modules[__name__], inspect.isclass)
    
    # Filter out classes that inherit from _ConjugatePair but are not _ConjugatePair itself
    conjugate_pairs = [cls for _, cls in all_classes if issubclass(cls, _ConjugatePair) and cls != _ConjugatePair]
    
    return conjugate_pairs
        
class _ConjugatePair(ABC):
    """ Abstract base class for conjugate pairs used in the Conjugate sampler. """

    def __init__(self, target):
        self.target = target

    @abstractmethod
    def validate_target(self):
        """ Validate the target distribution for the conjugate pair. """
        pass

    @abstractmethod
    def sample(self):
        """ Sample from the conjugate distribution. """
        pass


class _GaussianGammaPair(_ConjugatePair):
    """ Implementation for the Gaussian-Gamma conjugate pair."""

    def validate_target(self):
        if not isinstance(self.target.likelihood.distribution, (Gaussian, GMRF)):
            raise ValueError("Conjugate sampler only works with a Gaussian likelihood function")

        if not isinstance(self.target.prior, Gamma):
            raise ValueError("Conjugate sampler only works with Gamma prior")

        if self.target.prior.dim != 1:
            raise ValueError("Conjugate sampler only works with univariate Gamma prior")

        key, value = _get_conjugate_parameter(self.target)
        if key == "cov":
            if not _check_conjugate_parameter_is_scalar_reciprocal(value):
                raise ValueError("Gaussian-Gamma conjugate pair defined via covariance requires `cov` for the `Gaussian` to be: lambda x : 1.0/x for the conjugate parameter")
        elif key == "prec":
            if not _check_conjugate_parameter_is_scalar_identity(value):
                raise ValueError("Gaussian-Gamma conjugate pair defined via precision requires `prec` for the `Gaussian` to be: lambda x : x for the conjugate parameter")
        else:
            raise ValueError("Conjugate sampler only works with Gaussian likelihood functions where conjugate parameter is defined via covariance or precision")

    def sample(self):
        # Extract variables
        b = self.target.likelihood.data                                 # mu
        m = len(b)                                                      # n
        Ax = self.target.likelihood.distribution.mean                   # x_i
        L = self.target.likelihood.distribution(np.array([1])).sqrtprec # L
        alpha = self.target.prior.shape                                 # alpha
        beta = self.target.prior.rate                                   # beta

        # Create Gamma distribution and sample
        dist = Gamma(shape=m/2 + alpha, rate=.5 * np.linalg.norm(L @ (Ax - b))**2 + beta)

        return dist.sample()


class _RegularizedGaussianGammaPair(_ConjugatePair):
    """Implementation for the Regularized Gaussian-Gamma conjugate pair."""

    def validate_target(self):
        if not isinstance(self.target.likelihood.distribution, (RegularizedGaussian, RegularizedGMRF)):
            raise ValueError("Conjugate sampler only works with a Regularized Gaussian likelihood function")

        if not isinstance(self.target.prior, Gamma):
            raise ValueError("Conjugate sampler only works with Gamma prior")

        if self.target.prior.dim != 1:
            raise ValueError("Conjugate sampler only works with univariate Gamma prior")

        if self.target.likelihood.distribution.preset not in ["nonnegativity"]:
            raise ValueError("Conjugate sampler only works with implicit regularized Gaussian likelihood with nonnegativity constraints")

        key, value = _get_conjugate_parameter(self.target)
        if key == "cov":
            if not _check_conjugate_parameter_is_scalar_reciprocal(value):
                raise ValueError("Regularized Gaussian-Gamma conjugate pair defined via covariance requires cov: lambda x : 1.0/x for the conjugate parameter")
        elif key == "prec":
            if not _check_conjugate_parameter_is_scalar_identity(value):
                raise ValueError("Regularized Gaussian-Gamma conjugate pair defined via precision requires prec: lambda x : x for the conjugate parameter")
        else:
            raise ValueError("Conjugate sampler only works with Regularized Gaussian likelihood functions where conjugate parameter is defined via covariance or precision")

    def sample(self):
        # Extract variables
        b = self.target.likelihood.data                                 # mu
        m = np.count_nonzero(b)                                         # n
        Ax = self.target.likelihood.distribution.mean                   # x_i
        L = self.target.likelihood.distribution(np.array([1])).sqrtprec # L
        alpha = self.target.prior.shape                                 # alpha
        beta = self.target.prior.rate                                   # beta

        # Create Gamma distribution and sample
        dist = Gamma(shape=m/2 + alpha, rate=.5 * np.linalg.norm(L @ (Ax - b))**2 + beta)

        return dist.sample()

def _get_conjugate_parameter(target):
    """Extract the parameter name, attribute and variable location of the conjugate parameter."""
    par_name = target.prior.name
    mutable_likelihood_vars  = target.likelihood.distribution.get_mutable_variables()

    for var_key in mutable_likelihood_vars:
        attr = getattr(target.likelihood.distribution, var_key)
        if callable(attr) and par_name in get_non_default_args(attr):
            return var_key, attr
    raise ValueError(f"Unable to find conjugate parameter {par_name} in likelihood function for conjugate sampler with target {target}")

def _check_conjugate_parameter_is_scalar_identity(f):
    """Tests whether a function (scalar to scalar) is the identity (lambda x : x)."""
    return all(f(x) == x for x in [1.0, 10.0, 100.0])

def _check_conjugate_parameter_is_scalar_reciprocal(f):
    """Tests whether a function (scalar to scalar) is the reciprocal (lambda x : 1.0/x)."""
    return all(math.isclose(f(x), 1.0 / x) for x in [1.0, 10.0, 100.0])
