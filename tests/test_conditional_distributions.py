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
        2) that the cov @property setter is invoked after setting cov directly (here _cov was None).
    """
    model = cuqi.model.Model(lambda x:x, 1, 1) #Simple 1 par model
    dist = cuqi.distribution.GaussianCov(mean=model)
    likelihood = dist.to_likelihood(5)
    likelihood.log(10, 5)
