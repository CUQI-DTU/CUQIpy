import cuqi
import pytest

#TODO. Make tests for all distributions going through their input variables
@pytest.mark.parametrize("dist, expected",[
    (cuqi.distribution.GaussianCov(), ["mean", "cov"]),
    (cuqi.distribution.GaussianCov(mean=1), ["cov"]),
    (cuqi.distribution.GaussianCov(cov=1), ["mean"]),
    (cuqi.distribution.GaussianCov(mean=lambda m: m, cov=lambda c:c), ["m", "c"]),
    (cuqi.distribution.GaussianCov(mean=1, cov=lambda c:c), ["c"]),
    (cuqi.distribution.GaussianCov(mean=lambda m:m, cov=1), ["m"]),
])
def test_conditional_parameters(dist, expected):
    assert dist.get_conditioning_variables() == expected

def test_conditioning_through_likelihood(): #TODO. Add more dists to test here!
    """ This checks the flow of likelihood evaluation. In particular it checks:
        1) that the mean @property setter is invoked after model evaluation through function/model.
        2) that the cov @property setter is invoked after setting cov directly (here cov getter returned None).
    """
    model = cuqi.model.Model(lambda x:x, 1, 1) #Simple 1 par model
    dist = cuqi.distribution.GaussianCov(mean=model)
    likelihood = dist.to_likelihood(5)
    likelihood.log(10, 5)
    assert(likelihood.get_parameter_names() == ['cov', 'x'])

def test_conditional_wrong_keyword():
    """ This checks if we raise error if keyword is not a conditional variable. """
    x = cuqi.distribution.GaussianCov(mean=1)
    with pytest.raises(ValueError):
        x(name="hey") #Should expect value-error since name not conditional variable.

def test_conditional_both_args_kwargs():
    """ This tests that we throw error if we accidentally provide arg and kwarg for same variable. """
    x = cuqi.distribution.GaussianCov(mean=1)
    with pytest.raises(ValueError):
        x(1, cov=1) #Should expect value-error since cov is given as arg and kwarg.

def test_conditional_multiple_args():
    """ This tests if conditional variables are printed correctly if they appear in multiple mutable variables"""
    dist = cuqi.distribution.GaussianCov(mean=lambda x,y: x+y, cov=lambda y,z: y+z)
    assert dist.get_conditioning_variables() == ["x","y","z"]

def test_conditional_partial_function():
    """ This tests the partial functions we define if only part of a callable is given. """
    dist = cuqi.distribution.GaussianCov(mean=lambda x,y: x+y, cov=lambda y,z: y+z)
    dist2 = dist(x=1, y=2)
    assert dist2.get_conditioning_variables() == ["z"]
    assert dist2.mean == 3
    assert cuqi.utilities.get_non_default_args(dist2.cov) == ["z"]
