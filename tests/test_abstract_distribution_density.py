import cuqi
import pytest


#TODO. Make tests for all distributions going through their input variables
@pytest.mark.parametrize("dist, expected",[
    (cuqi.distribution.Gaussian(), ["mean", "cov"]),
    (cuqi.distribution.Gaussian(mean=1), ["cov"]),
    (cuqi.distribution.Gaussian(cov=1), ["mean"]),
    (cuqi.distribution.Gaussian(mean=lambda m: m, cov=lambda c:c), ["m", "c"]),
    (cuqi.distribution.Gaussian(mean=1, cov=lambda c:c), ["c"]),
    (cuqi.distribution.Gaussian(mean=lambda m:m, cov=1), ["m"]),
])
def test_conditioning_variables(dist, expected):
    assert dist.get_conditioning_variables() == expected

def test_conditioning_through_likelihood(): #TODO. Add more dists to test here!
    """ This checks the flow of likelihood evaluation. In particular it checks:
        1) that the mean @property setter is invoked after model evaluation through function/model.
        2) that the cov @property setter is invoked after setting cov directly (here cov getter returned None).
    """
    model = cuqi.model.Model(lambda x:x, 1, 1) #Simple 1 par model
    dist = cuqi.distribution.Gaussian(mean=model)
    likelihood = dist.to_likelihood(5)
    likelihood.logd(10, 5)
    assert(likelihood.get_parameter_names() == ['cov', 'x'])

def test_conditioning_wrong_keyword():
    """ This checks if we raise error if keyword is not a conditioning variable. """
    x = cuqi.distribution.Gaussian(mean=1)
    with pytest.raises(ValueError):
        x(name="hey") #Should expect value-error since name not conditioning variable.

def test_conditioning_arg_as_mutable_var():
    """ This checks if we raise error if 2 args are given for a distribution that has no conditioning variables. """
    x = cuqi.distribution.Gaussian(mean=1, cov=1)
    with pytest.raises(ValueError):
        x(5, 3) #Should expect value-error since no cond vars

def test_conditioning_on_main_parameter():
    """ This checks if we can condition on the main parameter in various ways. """
    x = cuqi.distribution.Gaussian()

    # With keywords (name also automatically inferred)
    assert isinstance(x(mean=1, x=5), cuqi.likelihood.Likelihood)
    assert isinstance(x(mean=1, cov=1, x=5), cuqi.density.EvaluatedDensity)
    
    # Now using positional arguments
    assert isinstance(x(1, 1, 5), cuqi.density.EvaluatedDensity)

def test_conditioning_kwarg_as_mutable_var():
    """ This checks if we allow kwargs for a distribution that has no conditioning variables. """
    x = cuqi.distribution.Gaussian(mean=1, cov=1)
    x = x(cov=2) #This should be ok and not throw an error
    assert x.cov == 2

def test_conditioning_both_args_kwargs():
    """ This tests that we throw error if we accidentally provide arg and kwarg for same variable. """
    x = cuqi.distribution.Gaussian(mean=1)
    with pytest.raises(ValueError):
        x(1, cov=1) #Should expect value-error since cov is given as arg and kwarg.

def test_conditioning_multiple_args():
    """ This tests if conditional variables are printed correctly if they appear in multiple mutable variables"""
    dist = cuqi.distribution.Gaussian(mean=lambda x,y: x+y, cov=lambda y,z: y+z)
    assert dist.get_conditioning_variables() == ["x","y","z"]

def test_conditioning_partial_function():
    """ This tests the partial functions we define if only part of a callable is given. """
    dist = cuqi.distribution.Gaussian(mean=lambda x,y: x+y, cov=lambda y,z: y+z)
    dist2 = dist(x=1, y=2)
    assert dist2.get_conditioning_variables() == ["z"]
    assert dist2.mean == 3
    assert cuqi.utilities.get_non_default_args(dist2.cov) == ["z"]

def test_conditioning_keeps_name():
    """ This tests if the name of the distribution is kept when conditioning. """
    y = cuqi.distribution.Gaussian(lambda x:x)
    
    assert y(x=1, cov=1).name == y.name
    assert y(x=1)(cov=1).name == y.name
    assert y(x=1)(cov=1)().name == y.name

    assert y(x=1).name == y.name
    assert y(cov=1).name == y.name
    assert y().name == y.name

def test_conditioning_class_flow():
    """ This tests the class flow of conditioning a conditional distribution in various ways """

    # Initial conditional distribution on parameter x and cov.
    y = cuqi.distribution.Gaussian(lambda x:x)

    # Conditioning on x is still conditional on cov
    assert y(x=1).is_cond

    # Conditioning on y (name of distribution) should return a likelihood
    assert isinstance(y(y=1), cuqi.likelihood.Likelihood)

    # Conditioning on x and cov be a regular distribution
    assert isinstance(y(x=1, cov=1), cuqi.distribution.Distribution)

    # Conditioning on all unspecified variables should return constant density
    assert isinstance(y(x=1, y=1, cov=1), cuqi.density.EvaluatedDensity)

    # Conditioning on all parameters in various ways should also return constant density
    # (in between they might change to Likelihood)
    assert isinstance(y(x=1)(y=1)(cov=1), cuqi.density.EvaluatedDensity)
    assert isinstance(y(x=1)(cov=1)(y=1), cuqi.density.EvaluatedDensity)
    assert isinstance(y(y=1)(x=1)(cov=1), cuqi.density.EvaluatedDensity)
    assert isinstance(y(y=1)(cov=1)(x=1), cuqi.density.EvaluatedDensity)
    assert isinstance(y(cov=1)(x=1)(y=1), cuqi.density.EvaluatedDensity)
    assert isinstance(y(cov=1)(y=1)(x=1), cuqi.density.EvaluatedDensity)

def test_logp_conditional():
    """ This tests logp evaluation for conditional distributions """
    # Base example logp value
    true_val = cuqi.distribution.Gaussian(3, 7).logd(13)

    # Distribution with no specified parameters
    x = cuqi.distribution.Gaussian(cov=lambda s:s)

    # Test logp evaluates correctly in various cases
    assert x.logd(mean=3, s=7, x=13) == true_val
    assert x(x=13).logd(mean=3, s=7) == true_val
    assert x(x=13, mean=3).logd(s=7) == true_val
    assert x(x=13, mean=3, s=7).logd() == true_val
    assert x(mean=3).logd(s=7, x=13) == true_val
    assert x(mean=3, s=7).logd(x=13) == true_val
    assert x(mean=3, x=13).logd(s=7) == true_val

def test_logd_err_handling():
    """ This tests if logp correctly identifies errors in the input """
    x = cuqi.distribution.Gaussian(cov=lambda s:s)

    # Test that we raise error if we don't provide all parameters
    with pytest.raises(ValueError, match=r"To evaluate the log density all conditioning variables must be specified"):
        x.logd(x=3)

    # Test that we raise error if we provide parameters that are not specified
    with pytest.raises(ValueError, match=r"do not match keyword arguments"):
        x.logd(mean=3, s=7, x=13, y=1)

def test_cond_positional_and_kwargs():
    """ Test conditioning for both positional and kwargs """
    x = cuqi.distribution.Gaussian(cov=lambda s:s)

    logd = x(mean=3, cov=7).logd(13)

    # Conditioning full positional
    assert x(3, 7, 13).value == logd
    assert x(3, 7)(13).value == logd
    assert x(3)(7, 13).value == logd
    assert x(3)(7)(13).value == logd

    # Conditioning full kwargs
    assert x(mean=3, cov=7, x=13).value == logd
    assert x(mean=3, cov=7)(x=13).value == logd
    assert x(mean=3)(cov=7, x=13).value == logd
    assert x(mean=3)(cov=7)(x=13).value == logd

    # Conditioning partial positional
    assert x(3, s=7, x=13).value == logd
    assert x(3, 7, x=13).value == logd
    assert x(3, s=7)(13).value == logd
    assert x(mean=3)(7, x=13).value == logd
    assert x(mean=3)(7)(x=13).value == logd
    assert x(mean=3)(7)(13).value == logd

def test_logd_positional_and_kwargs():
    """ Test logd for both positional and kwargs """
    x = cuqi.distribution.Gaussian(cov=lambda s:s)

    logd = x(mean=3, s=7).logd(13)

    # logd full kwargs
    assert x.logd(mean=3, s=7, x=13) == logd

    # logd full positional
    assert x.logd(3, 7, 13) == logd

    # logd partial positional
    assert x.logd(3, s=7, x=13) == logd
    assert x.logd(3, 7, x=13) == logd



