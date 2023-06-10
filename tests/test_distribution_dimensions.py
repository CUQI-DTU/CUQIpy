import cuqi
import numpy as np
import pytest

dists = [
    cuqi.distribution.Beta,
    cuqi.distribution.Cauchy,
    cuqi.distribution.CMRF,
    cuqi.distribution.Gamma,
    cuqi.distribution.Gaussian,
    cuqi.distribution.GMRF,
    cuqi.distribution.InverseGamma,
    cuqi.distribution.LMRF,
    cuqi.distribution.Laplace,
    cuqi.distribution.Lognormal,
    cuqi.distribution.Normal,
    cuqi.distribution.Uniform
]

scalar_values = [1]

@pytest.mark.parametrize('dist', dists)
@pytest.mark.parametrize('scalar', scalar_values)
def test_multivariate_scalar_vars_logd(dist, scalar):
    """ Test logd method of a multivariate distribution with scalar parameters """

    if dist == cuqi.distribution.GMRF or dist == cuqi.distribution.Gamma:
        pytest.skip(f"{dist.__name__} does not handle scalar setup for logd yet.")

    dist1 = dist(scalar * np.ones(5), 9)
    dist2 = dist(scalar, 9, dim=5)

    # Check that the dists are not conditional still
    if getattr(dist1, 'is_cond', False):
        cond_vars = dist1.get_conditioning_variables()
        extra_vars = [10, 11, 12, 13, 14]
        dist1 = dist1(*extra_vars[:len(cond_vars)])
        dist2 = dist2(*extra_vars[:len(cond_vars)])

    val = np.random.randn(5)
    assert np.allclose(dist1.logd(val), dist2.logd(val))

@pytest.mark.parametrize('dist', dists)
@pytest.mark.parametrize('scalar', scalar_values)
def test_multivariate_scalar_vars_sample(dist, scalar):
    """ Test sample method of a multivariate distribution with scalar parameters """

    if dist == cuqi.distribution.GMRF or dist == cuqi.distribution.Gamma or dist == cuqi.distribution.Lognormal:
        pytest.skip(f"{dist.__name__} does not handle scalar with sample yet.")

    dist1 = dist(scalar * np.ones(5), 9)
    dist2 = dist(scalar, 9, dim=5)

    # Check that the dists are not conditional still
    if getattr(dist1, 'is_cond', False):
        cond_vars = dist1.get_conditioning_variables()
        extra_vars = [10, 11, 12, 13, 14]
        dist1 = dist1(*extra_vars[:len(cond_vars)])
        dist2 = dist2(*extra_vars[:len(cond_vars)])

    try:
        assert np.allclose(dist1.sample(rng=np.random.RandomState(0)), dist2.sample(rng=np.random.RandomState(0)))
    except NotImplementedError:
        pass

@pytest.mark.parametrize('dist', dists)
@pytest.mark.parametrize('scalar', scalar_values)
def test_multivariate_scalar_vars_gradient(dist, scalar):
    """ Test gradient method of a multivariate distribution with scalar parameters """

    if dist == cuqi.distribution.GMRF or dist == cuqi.distribution.Gamma or dist == cuqi.distribution.Lognormal:
        pytest.skip(f"{dist.__name__} does not handle scalar setup for gradient yet.")

    dist1 = dist(scalar * np.ones(5), 9)
    dist2 = dist(scalar, 9, dim=5)

    # Check that the dists are not conditional still
    if getattr(dist1, 'is_cond', False):
        cond_vars = dist1.get_conditioning_variables()
        extra_vars = [10, 11, 12, 13, 14]
        dist1 = dist1(*extra_vars[:len(cond_vars)])
        dist2 = dist2(*extra_vars[:len(cond_vars)])

    val = 0.2 * np.ones(5)
    if dist == cuqi.distribution.InverseGamma:
        pytest.skip(f"{dist.__name__} Skip inverse Gamma for now.")
    try:
        assert np.allclose(dist1.gradient(val), dist2.gradient(val))
    except NotImplementedError:
            pass
