import numpy as np
from abc import ABC, abstractmethod
import math
from cuqi.experimental.mcmc import Sampler
from cuqi.distribution import Posterior, Gaussian, Gamma, GMRF, ModifiedHalfNormal
from cuqi.implicitprior import RegularizedGaussian, RegularizedGMRF, RegularizedUnboundedUniform
from cuqi.utilities import get_non_default_args, count_nonzero, count_constant_components_1D, count_constant_components_2D
from cuqi.geometry import Continuous1D, Continuous2D, Image2D

class Conjugate(Sampler):
    """ Conjugate sampler

    Sampler for sampling a posterior distribution which is a so-called "conjugate" distribution, i.e., where the likelihood and prior are conjugate to each other - denoted as a conjugate pair.   

    Currently supported conjugate pairs are:
    - (Gaussian, Gamma) where Gamma is defined on the precision parameter of the Gaussian
    - (GMRF, Gamma) where Gamma is defined on the precision parameter of the GMRF
    - (RegularizedGaussian, Gamma) with preset constraints only and Gamma is defined on the precision parameter of the RegularizedGaussian
    - (RegularizedGMRF, Gamma) with preset constraints only and Gamma is defined on the precision parameter of the RegularizedGMRF
    - (RegularizedGaussian, ModifiedHalfNormal) with preset constraints and regularization only
    - (RegularizedGMRF, ModifiedHalfNormal) with preset constraints and regularization only

    Currently the Gamma and ModifiedHalfNormal distribution must be univariate.

    A conjugate pair defines implicitly a so-called conjugate distribution which can be sampled from directly.

    The conjugate parameter is the parameter that both the likelihood and prior PDF depend on.

    For more information on conjugacy and conjugate distributions see https://en.wikipedia.org/wiki/Conjugate_prior.

    For implicit regularized Gaussians and the corresponding conjugacy relations, see:
    
    Section 3.3 from [1] Everink, Jasper M., Yiqiu Dong, and Martin S. Andersen. "Bayesian inference with projected densities." SIAM/ASA Journal on Uncertainty Quantification 11.3 (2023): 1025-1043.
    Section 4 from [2] Everink, Jasper M., Yiqiu Dong, and Martin S. Andersen. "Sparse Bayesian inference with regularized Gaussian distributions." Inverse Problems 39.11 (2023): 115004.

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
        elif isinstance(self.target.likelihood.distribution, RegularizedUnboundedUniform) and isinstance(self.target.prior, Gamma):
            # Check RegularizedUnboundedUniform before RegularizedGaussian and RegularizedGMRF due to the first inheriting from the second.
            self._conjugatepair = _RegularizedUnboundedUniformGammaPair(self.target)
        elif isinstance(self.target.likelihood.distribution, (RegularizedGaussian, RegularizedGMRF)) and isinstance(self.target.prior, Gamma):
            self._conjugatepair = _RegularizedGaussianGammaPair(self.target)
        elif isinstance(self.target.likelihood.distribution, (RegularizedGaussian, RegularizedGMRF)) and isinstance(self.target.prior, ModifiedHalfNormal):
            self._conjugatepair = _RegularizedGaussianModifiedHalfNormalPair(self.target)
        else:
            raise ValueError(f"Conjugacy is not defined for likelihood {type(self.target.likelihood.distribution)} and prior {type(self.target.prior)}, in CUQIpy")

    def conjugate_distribution(self):
        return self._conjugatepair.conjugate_distribution()

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
    def conjugate_distribution(self):
        """ Returns the posterior distribution in the form of a CUQIpy distribution """
        pass

    def sample(self):
        """ Sample from the conjugate distribution. """
        return self.conjugate_distribution().sample()


class _GaussianGammaPair(_ConjugatePair):
    """ Implementation for the Gaussian-Gamma conjugate pair."""

    def validate_target(self):
        if self.target.prior.dim != 1:
            raise ValueError("Gaussian-Gamma conjugacy only works with univariate Gamma prior")

        key_value_pairs = _get_conjugate_parameter(self.target)
        if len(key_value_pairs) != 1:
            raise ValueError(f"Multiple references to conjugate parameter {self.target.prior.name} found in likelihood. Only one occurance is supported.")
        for key, value in key_value_pairs:
            if key == "cov":
                if not _check_conjugate_parameter_is_scalar_linear_reciprocal(value):
                    raise ValueError("Gaussian-Gamma conjugate pair defined via covariance requires cov: lambda x : s/x for the conjugate parameter")
            elif key == "prec":
                if not _check_conjugate_parameter_is_scalar_linear(value):
                    raise ValueError("Gaussian-Gamma conjugate pair defined via precision requires prec: lambda x : s*x for the conjugate parameter")
            else:
                raise ValueError(f"RegularizedGaussian-ModifiedHalfNormal conjugacy does not support the conjugate parameter {self.target.prior.name} in the {key} attribute. Only cov and prec")

    def conjugate_distribution(self):
        # Extract variables
        b = self.target.likelihood.data                                 # mu
        m = len(b)                                                      # n
        Ax = self.target.likelihood.distribution.mean                   # x_i
        L = self.target.likelihood.distribution(np.array([1])).sqrtprec # L
        alpha = self.target.prior.shape                                 # alpha
        beta = self.target.prior.rate                                   # beta

        # Create Gamma distribution and sample
        return Gamma(shape=m/2 + alpha, rate=.5 * np.linalg.norm(L @ (Ax - b))**2 + beta)


class _RegularizedGaussianGammaPair(_ConjugatePair):
    """Implementation for the Regularized Gaussian-Gamma conjugate pair using the conjugacy rules from [1], Section 3.3."""

    def validate_target(self):
        if self.target.prior.dim != 1:
            raise ValueError("RegularizedGaussian-Gamma conjugacy only works with univariate ModifiedHalfNormal prior")

        if self.target.likelihood.distribution.preset["constraint"] not in ["nonnegativity"]:
            raise ValueError("RegularizedGaussian-Gamma conjugacy only works with implicit regularized Gaussian likelihood with nonnegativity constraints")

        key_value_pairs = _get_conjugate_parameter(self.target)
        if len(key_value_pairs) != 1:
            raise ValueError(f"Multiple references to conjugate parameter {self.target.prior.name} found in likelihood. Only one occurance is supported.")
        for key, value in key_value_pairs:
            if key == "cov":
                if not _check_conjugate_parameter_is_scalar_linear_reciprocal(value):
                    raise ValueError("Regularized Gaussian-Gamma conjugacy defined via covariance requires cov: lambda x : s/x for the conjugate parameter")
            elif key == "prec":
                if not _check_conjugate_parameter_is_scalar_linear(value):
                    raise ValueError("Regularized Gaussian-Gamma conjugacy defined via precision requires prec: lambda x : s*x for the conjugate parameter")
            else:
                raise ValueError(f"RegularizedGaussian-ModifiedHalfNormal conjugacy does not support the conjugate parameter {self.target.prior.name} in the {key} attribute. Only cov and prec")

    def conjugate_distribution(self):
        # Extract variables
        b = self.target.likelihood.data                                 # mu
        m = np.count_nonzero(b)                                         # n
        Ax = self.target.likelihood.distribution.mean                   # x_i
        L = self.target.likelihood.distribution(np.array([1])).sqrtprec # L
        alpha = self.target.prior.shape                                 # alpha
        beta = self.target.prior.rate                                   # beta

        # Create Gamma distribution and sample
        return Gamma(shape=m/2 + alpha, rate=.5 * np.linalg.norm(L @ (Ax - b))**2 + beta)
    
    
class _RegularizedUnboundedUniformGammaPair(_ConjugatePair):
    """Implementation for the RegularizedUnboundedUniform-ModifiedHalfNormal conjugate pair using the conjugacy rules from [2], Section 4."""

    def validate_target(self):
        if self.target.prior.dim != 1:
            raise ValueError("RegularizedUnboundedUniform-Gamma conjugacy only works with univariate Gamma prior")
        
        if self.target.likelihood.distribution.preset["regularization"] not in ["l1", "tv"]:
            raise ValueError("RegularizedUnboundedUniform-Gamma conjugacy only works with implicit regularized Gaussian likelihood with l1 or tv regularization")
        
        key_value_pairs = _get_conjugate_parameter(self.target)
        if len(key_value_pairs) != 1:
            raise ValueError(f"Multiple references to conjugate parameter {self.target.prior.name} found in likelihood. Only one occurance is supported.")
        for key, value in key_value_pairs:
            if key == "strength":
                if not _check_conjugate_parameter_is_scalar_linear(value):
                    raise ValueError("RegularizedUnboundedUniform-Gamma conjugacy defined via strength requires strength: lambda x : s*x for the conjugate parameter")
            else:
                raise ValueError(f"RegularizedUnboundedUniform-Gamma conjugacy does not support the conjugate parameter {self.target.prior.name} in the {key} attribute. Only strength is supported")

    def conjugate_distribution(self):
        # Extract prior variables
        alpha = self.target.prior.shape
        beta = self.target.prior.rate

        # Compute likelihood quantities
        x = self.target.likelihood.data
        m = _compute_sparsity_level(self.target)

        reg_op = self.target.likelihood.distribution._regularization_oper
        reg_strength = self.target.likelihood.distribution(np.array([1])).strength
        fx = reg_strength*np.linalg.norm(reg_op@x, ord = 1)

        # Create Gamma distribution
        return Gamma(shape=m/2 + alpha, rate=fx + beta)
    
class _RegularizedGaussianModifiedHalfNormalPair(_ConjugatePair):
    """Implementation for the Regularized Gaussian-ModifiedHalfNormal conjugate pair using the conjugacy rules from [2], Section 4."""

    def validate_target(self):
        if self.target.prior.dim != 1:
            raise ValueError("RegularizedGaussian-ModifiedHalfNormal conjugacy only works with univariate ModifiedHalfNormal prior")
        
        if self.target.likelihood.distribution.preset["regularization"] not in ["l1", "tv"]:
            raise ValueError("RegularizedGaussian-ModifiedHalfNormal conjugacy only works with implicit regularized Gaussian likelihood with l1 or tv regularization")

        key_value_pairs = _get_conjugate_parameter(self.target)
        if len(key_value_pairs) != 2:
            raise ValueError(f"Incorrect number of references to conjugate parameter {self.target.prior.name} found in likelihood. Found {len(key_value_pairs)} times, but needs to occur in prec or cov, and in strength")
        for key, value in key_value_pairs:
            if key == "strength":
                if not _check_conjugate_parameter_is_scalar_linear(value):
                    raise ValueError("RegularizedGaussian-ModifiedHalfNormal conjugacy defined via strength requires strength: lambda x : s*x for the conjugate parameter")
            elif key == "prec":
                if not _check_conjugate_parameter_is_scalar_quadratic(value):
                    raise ValueError("RegularizedGaussian-ModifiedHalfNormal conjugacy defined via precision requires prec: lambda x : s*x for the conjugate parameter")
            elif key == "cov":
                if not _check_conjugate_parameter_is_scalar_quadratic_reciprocal(value):
                    raise ValueError("RegularizedGaussian-ModifiedHalfNormal conjugacy  defined via covariance requires cov: lambda x : s/x for the conjugate parameter")
            else:
                raise ValueError(f"RegularizedGaussian-ModifiedHalfNormal conjugacy does not support the conjugate parameter {self.target.prior.name} in the {key} attribute. Only cov, prec and strength are supported")


    def conjugate_distribution(self):
        # Extract prior variables
        alpha = self.target.prior.alpha
        beta = self.target.prior.beta
        gamma = self.target.prior.gamma

        # Compute likelihood variables
        x = self.target.likelihood.data
        mu = self.target.likelihood.distribution.mean
        L = self.target.likelihood.distribution(np.array([1])).sqrtprec
        
        m = _compute_sparsity_level(self.target)

        reg_op = self.target.likelihood.distribution._regularization_oper
        reg_strength = self.target.likelihood.distribution(np.array([1])).strength
        fx = reg_strength*np.linalg.norm(reg_op@x, ord = 1)

        # Compute parameters of conjugate distribution
        conj_alpha = m + alpha
        conj_beta = 0.5*np.linalg.norm(L @ (mu - x))**2 + beta
        conj_gamma = -fx + gamma

        # Create conjugate distribution
        return ModifiedHalfNormal(conj_alpha, conj_beta, conj_gamma)
    

def _compute_sparsity_level(target):
    """Computes the sparsity level in accordance with Section 4 from [2],"""
    x = target.likelihood.data
    if target.likelihood.distribution.preset["constraint"] == "nonnegativity":
        if target.likelihood.distribution.preset["regularization"] == "l1":
            m = count_nonzero(x)
        elif target.likelihood.distribution.preset["regularization"] == "tv" and isinstance(target.likelihood.distribution.geometry, Continuous1D):
            m = count_constant_components_1D(x, lower = 0.0)
        elif target.likelihood.distribution.preset["regularization"] == "tv" and isinstance(target.likelihood.distribution.geometry, (Continuous2D, Image2D)):
            m = count_constant_components_2D(target.likelihood.distribution.geometry.par2fun(x), lower = 0.0)
    else: # No constraints, only regularization
        if target.likelihood.distribution.preset["regularization"] == "l1":
            m = count_nonzero(x)
        elif target.likelihood.distribution.preset["regularization"] == "tv" and isinstance(target.likelihood.distribution.geometry, Continuous1D):
            m = count_constant_components_1D(x)
        elif target.likelihood.distribution.preset["regularization"] == "tv" and isinstance(target.likelihood.distribution.geometry, (Continuous2D, Image2D)):
            m = count_constant_components_2D(target.likelihood.distribution.geometry.par2fun(x))
    return m


def _get_conjugate_parameter(target):
    """Extract the conjugate parameter name (e.g. d), and returns the mutable variable that is defined by the conjugate parameter, e.g. cov and its value e.g. lambda d:1/d"""
    par_name = target.prior.name
    mutable_likelihood_vars  = target.likelihood.distribution.get_mutable_variables()

    found_parameter_pairs = []

    for var_key in mutable_likelihood_vars:
        attr = getattr(target.likelihood.distribution, var_key)
        if callable(attr) and par_name in get_non_default_args(attr):
            found_parameter_pairs.append((var_key, attr))
    if len(found_parameter_pairs) == 0:
        raise ValueError(f"Unable to find conjugate parameter {par_name} in likelihood function for conjugate sampler with target {target}")
    return found_parameter_pairs

def _check_conjugate_parameter_is_scalar_identity(f):
    """Tests whether a function (scalar to scalar) is the identity (lambda x: x)."""
    test_values = [1.0, 10.0, 100.0]
    return all(np.allclose(f(x), x) for x in test_values)

def _check_conjugate_parameter_is_scalar_reciprocal(f):
    """Tests whether a function (scalar to scalar) is the reciprocal (lambda x : 1.0/x)."""
    return all(math.isclose(f(x), 1.0 / x) for x in [1.0, 10.0, 100.0])

def _check_conjugate_parameter_is_scalar_linear(f):
    """
        Tests whether a function (scalar to scalar) is linear (lambda x: s*x for some s).
        The tests checks whether the function is zero and some finite differences are constant.
    """
    test_values = [1.0, 10.0, 100.0]
    h = 1e-2
    finite_diffs = [(f(x + h*x)-f(x))/(h*x) for x in test_values]
    return np.isclose(f(0.0), 0.0) and all(np.allclose(c, finite_diffs[0]) for c in finite_diffs[1:])

def _check_conjugate_parameter_is_scalar_linear_reciprocal(f):
    """
        Tests whether a function (scalar to scalar) is a constant times the inverse of the input (lambda x: s/x for some s).
        The tests checks whether the the reciprocal of the function has constant finite differences.
    """
    g = lambda x : 1.0/f(x)
    test_values = [1.0, 10.0, 100.0]
    h = 1e-2
    finite_diffs = [(g(x + h*x)-g(x))/(h*x) for x in test_values]
    return all(np.allclose(c, finite_diffs[0]) for c in finite_diffs[1:])

def _check_conjugate_parameter_is_scalar_quadratic(f):
    """
        Tests whether a function (scalar to scalar) is linear (lambda x: s*x**2 for some s).
        The tests checks whether the function divided by the parameter is linear
    """
    return _check_conjugate_parameter_is_scalar_linear(lambda x: f(x)/x if x != 0.0 else f(0.0))

def _check_conjugate_parameter_is_scalar_quadratic_reciprocal(f):
    """
        Tests whether a function (scalar to scalar) is linear (lambda x: s*x**-2 for some s).
        The tests checks whether the function divided by the parameter is the reciprical of a linear function.
    """
    return _check_conjugate_parameter_is_scalar_linear_reciprocal(lambda x: f(x)/x)
