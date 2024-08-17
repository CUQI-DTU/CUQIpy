import numpy as np
from abc import ABC, abstractmethod
import math
from cuqi.experimental.mcmc import Sampler
from cuqi.distribution import Posterior, Gaussian, Gamma, GMRF
from cuqi.implicitprior import RegularizedGaussian, RegularizedGMRF
from cuqi.utilities import get_non_default_args

class Conjugate(Sampler):
    """ Conjugate sampler

    Sampler for sampling a posterior distribution which is a so-called "conjugate" distribution, i.e., where the likelihood and prior are conjugate to each other - denoted as a conjugate pair.   

    Currently supported conjugate pairs are:
    - (Gaussian, Gamma) where Gamma is defined on the precision parameter of the Gaussian
    - (GMRF, Gamma) where Gamma is defined on the precision parameter of the GMRF
    - (RegularizedGaussian, Gamma) with nonnegativity constraints only and Gamma is defined on the precision parameter of the RegularizedGaussian
    - (RegularizedGMRF, Gamma) with nonnegativity constraints only and Gamma is defined on the precision parameter of the RegularizedGMRF

    Currently the Gamma distribution must be univariate.

    A conjugate pair defines implicitly a so-called conjugate distribution which can be sampled from directly.

    The conjugate parameter is the parameter that both the likelihood and prior PDF depend on.

    For more information on conjugacy and conjugate distributions see https://en.wikipedia.org/wiki/Conjugate_prior.

    For implicit regularized Gaussians see:
    
    [1] Everink, Jasper M., Yiqiu Dong, and Martin S. Andersen. "Bayesian inference with projected densities." SIAM/ASA Journal on Uncertainty Quantification 11.3 (2023): 1025-1043.

    """

    def _initialize(self):
        pass

    @Sampler.target.setter # Overwrite the target setter to set the conjugate pair
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
        return 1 # Returns acceptance rate of 1

    def tune(self, skip_len, update_count):
        pass # No tuning required for conjugate sampler

    def _ensure_target_is_posterior(self):
        """ Ensure that the target is a Posterior distribution. """
        if not isinstance(self.target, Posterior):
            raise TypeError("Conjugate sampler requires a target of type Posterior")

    def _set_conjugatepair(self):
        """ Set the conjugate pair based on the likelihood and prior. This requires target to be set. """
        self._ensure_target_is_posterior()
        if isinstance(self.target.likelihood.distribution, (Gaussian, GMRF)) and isinstance(self.target.prior, Gamma):
            self._conjugatepair = _GaussianGammaPair(self.target)
        elif isinstance(self.target.likelihood.distribution, (RegularizedGaussian, RegularizedGMRF)) and isinstance(self.target.prior, Gamma):
            self._conjugatepair = _RegularizedGaussianGammaPair(self.target)
        else:
            raise ValueError(f"Conjugacy is not defined for likelihood {type(self.target.likelihood.distribution)} and prior {type(self.target.prior)}, in CUQIpy")

    def __repr__(self):
        msg = super().__repr__()
        if hasattr(self, "_conjugatepair"):
            msg += f"\n Conjugate pair:\n\t {type(self._conjugatepair).__name__.removeprefix('_')}"
        return msg
        
class _ConjugatePair(ABC):
    """ Abstract base class for conjugate pairs (likelihood, prior) used in the Conjugate sampler. """

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
            raise ValueError("Conjugate sampler for Gaussian likelihood functions only works when conjugate parameter is defined via covariance or precision")

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
            raise ValueError("Conjugate sampler for a Regularized Gaussian likelihood functions only works when conjugate parameter is defined via covariance or precision")

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
    """Extract the conjugate parameter name (e.g. d), and returns the mutable variable that is defined by the conjugate parameter, e.g. cov and its value e.g. lambda d:1/d"""
    par_name = target.prior.name
    mutable_likelihood_vars  = target.likelihood.distribution.get_mutable_variables()

    found_parameter_pairs = []

    for var_key in mutable_likelihood_vars:
        attr = getattr(target.likelihood.distribution, var_key)
        if callable(attr) and par_name in get_non_default_args(attr):
            found_parameter_pairs.append((var_key, attr))
    if len(found_parameter_pairs) == 1:
        return found_parameter_pairs[0]
    elif len(found_parameter_pairs) > 1:
        raise ValueError(f"Multiple references of parameter {par_name} found in likelihood function for conjugate sampler with target {target}. This is not supported.")
    else:
        raise ValueError(f"Unable to find conjugate parameter {par_name} in likelihood function for conjugate sampler with target {target}")

def _check_conjugate_parameter_is_scalar_identity(f):
    """Tests whether a function (scalar to scalar) is the identity (lambda x: x)."""
    test_values = [1.0, 10.0, 100.0]
    return all(np.allclose(f(x), x) for x in test_values)

def _check_conjugate_parameter_is_scalar_reciprocal(f):
    """Tests whether a function (scalar to scalar) is the reciprocal (lambda x : 1.0/x)."""
    return all(math.isclose(f(x), 1.0 / x) for x in [1.0, 10.0, 100.0])
