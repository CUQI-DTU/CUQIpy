# Test logic in posterior class. In particular when some distributions have default geometry

import pytest
import numpy as np
import cuqi
from .test_model import MultipleInputTestModel

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


def test_joint_distribution_with_multiple_inputs_model_reduce_to_posterior():
    """Test that a joint distribution based on a model with multiple inputs
    reduces to a posterior when data is observed and all inputs (except 1) are
    specified."""

    test_model = MultipleInputTestModel.helper_build_three_input_test_model()
    model = cuqi.model.Model(
        test_model.forward_map,
        gradient=test_model.gradient_form2,
        domain_geometry=test_model.domain_geometry,
        range_geometry=test_model.range_geometry,
    )

    # Create priors
    x_dist = cuqi.distribution.Gaussian(
        mean=np.zeros(3),
        cov=np.eye(3))
    y_dist = cuqi.distribution.Gaussian(
        mean=np.zeros(2),
        cov=np.eye(2))
    z_dist = cuqi.distribution.Gaussian(
        mean=np.zeros(3),
        cov=np.eye(3))

    # Create data distribution
    data_dist = cuqi.distribution.Gaussian(
        mean=model(x_dist, y_dist, z_dist), cov = 1.0)

    # Create likelihood
    likelihood = data_dist(data_dist = np.array([2,2,3]))

    # Create joint distribution
    posterior = cuqi.distribution.JointDistribution(
        likelihood,
        x_dist,
        y_dist,
        z_dist
    )

    assert isinstance(posterior, cuqi.distribution.JointDistribution)

    # This should reduce to a posterior object
    post_x_y = posterior(x_dist=np.array([1, 1, 1]), y_dist=np.array([0, 1]))

    assert isinstance(post_x_y, cuqi.distribution.Posterior)