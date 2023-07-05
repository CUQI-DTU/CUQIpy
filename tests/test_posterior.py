# Test logic in posterior class. In particular when some distributions have default geometry

import pytest
import numpy as np
import cuqi

def test_posterior_should_give_Model_priority_for_geometry():
    A = cuqi.model.Model(lambda x: x.ravel(), range_geometry=10**2, domain_geometry=cuqi.geometry.Image2D((10, 10)))
    x = cuqi.distribution.Gaussian(0, 1, geometry=10**2) # Default geometry
    y = cuqi.distribution.Gaussian(A(x), 1)
    posterior = cuqi.distribution.Posterior(y.to_likelihood(np.zeros(10**2)), x)
    assert type(posterior.geometry) == cuqi.geometry.Image2D
    assert posterior.geometry == A.domain_geometry

def test_posterior_should_give_prior_priority_for_geometry_if_default_model_geometry():
    A = cuqi.model.Model(lambda x: x.ravel(), range_geometry=10**2, domain_geometry=(10,10))
    x = cuqi.distribution.Gaussian(0, 1, geometry=cuqi.geometry.Image2D((10, 10)))
    y = cuqi.distribution.Gaussian(A(x), 1)
    posterior = cuqi.distribution.Posterior(y.to_likelihood(np.zeros(10**2)), x)
    assert type(posterior.geometry) == cuqi.geometry.Image2D
    assert posterior.geometry == x.geometry

def test_posterior_should_give_likelihood_priority_for_geometry_if_default_prior_geometry():
    x = cuqi.distribution.Gaussian(0, 1, geometry=(10,10))
    y = cuqi.likelihood.UserDefinedLikelihood(logpdf_func=lambda x: x, geometry=cuqi.geometry.Image2D((10, 10)))
    posterior = cuqi.distribution.Posterior(y, x)
    assert type(posterior.geometry) == cuqi.geometry.Image2D
    assert posterior.geometry == y.geometry

def test_posterior_should_not_allow_prior_wrong_shape_if_model_has_geometry():
    A = cuqi.model.Model(lambda x: x.ravel(), range_geometry=10**2, domain_geometry=cuqi.geometry.Image2D((10, 10)))
    x = cuqi.distribution.Gaussian(0, 1, geometry=cuqi.geometry.Continuous1D(10**2)) #1D shape
    y = cuqi.distribution.Gaussian(A(x), 1)

    with pytest.raises(ValueError):
        posterior = cuqi.distribution.Posterior(y.to_likelihood(np.zeros(10**2)), x)



