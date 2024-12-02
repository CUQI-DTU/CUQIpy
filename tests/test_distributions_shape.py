import cuqi
import pytest
import inspect
import numpy as np

# Do not run the consistency shape tests for the following distributions
ignore_list = [
    "JointDistribution",
    "_StackedJointDistribution",
    "MultipleLikelihoodPosterior",
    "UserDefinedDistribution",
    "DistributionGallery",
    "Posterior",
    "Distribution",
    "JointGaussianSqrtPrec",
]

# Define cases to skip (these are TODO)
skip_logd = [
    cuqi.distribution.Gamma, # Missing force_ndarray
]
skip_sample = [
    cuqi.distribution.Gamma, # Missing force_ndarray
    cuqi.distribution.ModifiedHalfNormal,
    cuqi.distribution.Lognormal,
]
skip_gradient = [
    cuqi.distribution.Gamma, # Missing force_ndarray
    cuqi.distribution.Lognormal,
    cuqi.distribution.InverseGamma,
]

# Get all classes in the cuqi.distribution module, and ignore the ones in ignore list
dists = [
    cls
    for name, cls in inspect.getmembers(cuqi.distribution, inspect.isclass)
    if name not in ignore_list
]


def prepare_dist(dist):
    """ Prepare the distribution for testing with scalar variables """

    # Define all parameters for the distributions
    dist_params = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    # Unpack the first two elements for initial distribution parameters
    first_param, second_param, *extra_params = dist_params

    # Define distributions
    dist_from_vec = dist(first_param * np.ones(5), second_param)
    dist_from_dim = dist(first_param, second_param, geometry=5)

    # If distribution is still conditional it needs more parameters
    if dist_from_vec.is_cond:
        cond_vars = dist_from_vec.get_conditioning_variables()
        dist_from_vec = dist_from_vec(*extra_params[: len(cond_vars)])
        dist_from_dim = dist_from_dim(*extra_params[: len(cond_vars)])

    return dist_from_vec, dist_from_dim


@pytest.mark.parametrize("dist", dists, ids=lambda d: d.__name__)
def test_multivariate_scalar_vars_logd(dist):
    if dist in skip_logd:
        pytest.skip(f"{dist.__name__} does not handle scalar setup for logd yet.")
    dist_from_vec, dist_from_dim = prepare_dist(dist)
    val = np.random.randn(5)
    assert np.allclose(
        dist_from_vec.logd(val),
        dist_from_dim.logd(val),
        equal_nan=True
    )


@pytest.mark.parametrize("dist", dists, ids=lambda d: d.__name__)
def test_multivariate_scalar_vars_sample(dist):
    if dist in skip_sample:
        pytest.skip(f"{dist.__name__} does not handle scalar with sample yet.")
    dist_from_vec, dist_from_dim = prepare_dist(dist)
    try:
        assert np.allclose(
            dist_from_vec.sample(rng=np.random.RandomState(0)),
            dist_from_dim.sample(rng=np.random.RandomState(0)),
        )
    except NotImplementedError:
        pass  # Pass the test if NotImplementedError is raised


@pytest.mark.parametrize("dist", dists, ids=lambda d: d.__name__)
def test_multivariate_scalar_vars_gradient(dist):
    if dist in skip_gradient:
        pytest.skip(f"{dist.__name__} does not handle scalar setup for gradient yet.")
    dist_from_vec, dist_from_dim = prepare_dist(dist)
    val = 0.2 * np.ones(5)
    try:
        assert np.allclose(
            dist_from_vec.gradient(val),
            dist_from_dim.gradient(val),
            equal_nan=True
        )
    except NotImplementedError:
        pass  # Pass the test if NotImplementedError is raised
