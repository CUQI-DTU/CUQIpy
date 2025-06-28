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


def test_joint_distribution_with_multiple_inputs_model_reduce_to_posterior():
    forward_map = lambda x, y, z: x * y[0] + z * y[1]

    # gradient with respect to x
    def gradient_x(direction, x, y, z):
        return direction * y[0]

    # gradient with respect to y
    def gradient_y(direction, x, y, z):
        return np.array([direction @ x, direction @ z])

    # gradient with respect to z
    def gradient_z(direction, x, y, z):
        return direction * y[1]

    # gradient with respect to all inputs (form 2, callable)
    gradient_form2 = (gradient_x, gradient_y, gradient_z)

    # Assign the gradient functions to the test model
    domain_geometry = (
        cuqi.geometry.Continuous1D(3),
        cuqi.geometry.Continuous1D(2),
        cuqi.geometry.Continuous1D(3),
    )
    range_geometry = cuqi.geometry.Continuous1D(3)

    model_class = cuqi.model.Model

    model = model_class(
        forward=forward_map,
        gradient=gradient_form2,
        domain_geometry=domain_geometry,
        range_geometry=range_geometry)

    data_dist = cuqi.distribution.Gaussian(
        mean=model, cov = 1.0)#, cov=lambda x:x)
    likelihood = data_dist(data_dist = np.array([2,2,3]))

    x = cuqi.distribution.Gaussian(
        mean=np.zeros(3),
        cov=np.eye(3))
    y = cuqi.distribution.Gaussian(
        mean=np.zeros(2),
        cov=np.eye(2))
    z = cuqi.distribution.Gaussian(
        mean=np.zeros(3),
        cov=np.eye(3)) 

    posterior = cuqi.distribution.JointDistribution(
        likelihood,
        x,
        y,
        z)

    assert isinstance(posterior, cuqi.distribution.JointDistribution)

    post_x_y = posterior(x=np.array([1, 1, 1]), y=np.array([0, 1]))

    assert isinstance(post_x_y, cuqi.distribution.Posterior)