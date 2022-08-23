import cuqi
import numpy as np
import pytest

def test_joint_dist_dim_geometry():
    """ Test the dimension and geometry properties of a joint distribution """

    model, data, _ = cuqi.testproblem.Poisson_1D.get_components() # Model is m times n, m != n.

    # Bayesian model
    d = cuqi.distribution.Gamma(1, 1e-4)
    x = cuqi.distribution.GaussianCov(np.zeros(model.domain_dim), lambda d: d)
    y = cuqi.distribution.GaussianCov(model, 1)

    # Joint distribution
    J = cuqi.distribution.JointDistribution([d, x, y])

    # Test the dimensions
    assert J.dim == [d.dim, x.dim, y.dim]

    # Test the geometries
    assert J.geometry == [d.geometry, x.geometry, y.geometry]

    # Test dim and geometry for posterior
    P = J(y=data)

    assert P.dim == [d.dim, x.dim]

    assert P.geometry == [d.geometry, x.geometry]


@pytest.mark.parametrize("densities", [
    [
        cuqi.distribution.Gamma(1, 1e-4, name="x"),
        cuqi.distribution.Normal(0, 1, name="y")
    ],
    [
        cuqi.distribution.Gamma(1, 1e-4, name="x"),
        cuqi.distribution.Normal(0, lambda x:x, name="y")
    ],
    [
        cuqi.distribution.Gamma(1, 1e-4, name="d"),
        cuqi.distribution.Gamma(1, 1e-2, name="l"),
        cuqi.distribution.GaussianCov(np.zeros(8), lambda d: d, name="x"),
        cuqi.distribution.GaussianCov(
            mean=cuqi.testproblem.Deconvolution1D(dim=8).model,
            cov=lambda l: l,
            name="y"
        )
    ],
    [
        cuqi.distribution.Normal(0, 1e-2, name="z"),
        cuqi.distribution.Gamma(1, lambda z: abs(z), name="d"),
        cuqi.distribution.Gamma(lambda z: z, 1e-2, name="l"),
        cuqi.distribution.GaussianCov(np.zeros(8), lambda d: d, name="x"),
        cuqi.distribution.GaussianCov(
            mean=cuqi.testproblem.Deconvolution1D(dim=8).model,
            cov=lambda l: l,
            name="y"
        )
    ],
])
def test_joint_dist_logd(densities):
    """ Tests the log density function of the Joint distribution """

    # Create a joint distribution
    J = cuqi.distribution.JointDistribution(densities)

    # Get the parameter names
    names = J.get_parameter_names()

    # Get dimensions
    dims = J.dim

    # Make dictionary of random parameters matching names and dimensions
    params = {name: abs(np.random.randn(dim)) for name, dim in zip(names, dims)}

    # Evaluate the log density function at random points
    logd = J.logd(**params)

    # Manually evaluate the log density function for each density
    logd_manual = 0
    for density in densities:
        # Extract the parameters for this density
        density_params = {name: params[name] for name in density.get_parameter_names()}
        logd_manual += density.logd(**density_params)

    # Compare the log density functions
    assert logd == pytest.approx(logd_manual)

def test_joint_dist_properties():
    """ This tests various properties of the joint distribution """

    densities = [
        cuqi.distribution.Normal(0, 1e-2, name="z"),
        cuqi.distribution.Gamma(1, lambda z: abs(z), name="d"),
        cuqi.distribution.Gamma(lambda z: z, 1e-2, name="l"),
        cuqi.distribution.GaussianCov(np.zeros(8), lambda d: d, name="x"),
        cuqi.distribution.GaussianCov(
            mean=cuqi.testproblem.Deconvolution1D(dim=8).model,
            cov=lambda l: l,
            name="y"
        )
    ]

    # Create a joint distribution
    J = cuqi.distribution.JointDistribution(densities)

    # Now check various properties. First check the dimension
    assert J.dim == [density.dim for density in densities]

    # Check the geometry
    assert J.geometry == [density.geometry for density in densities]

    # Check the parameter names
    assert J.get_parameter_names() == [density.name for density in densities]

    # Check list of distributions
    assert J.distributions == densities

    # Check likelihoods
    assert J.likelihoods == []

    # Now we condition y on some data
    data = cuqi.testproblem.Deconvolution1D(dim=8).data

    P = J(y=data)

    # Check the dimension
    assert P.dim == [density.dim for density in densities[:-1]]

    # Check the geometry
    assert P.geometry == [density.geometry for density in densities[:-1]]

    # Check the parameter names
    assert P.get_parameter_names() == [density.name for density in densities[:-1]]

    # Check list of distributions by comparing the names
    assert [dist.name for dist in P.distributions] == [density.name for density in densities[:-1]]

    # Check likelihoods by comparing the names
    assert [L.name for L in P.likelihoods] == [densities[-1].name]

def test_joint_dist_reduce():
    """ This tests the reduce hack for the joint distribution. """

    densities = [
        cuqi.distribution.Normal(0, 1e-2, name="z"),
        cuqi.distribution.Gamma(1, lambda z: abs(z), name="d"),
        cuqi.distribution.Gamma(lambda z: z, 1e-2, name="l"),
        cuqi.distribution.GaussianCov(np.zeros(8), lambda d: d, name="x"),
        cuqi.distribution.GaussianCov(
            mean=cuqi.testproblem.Deconvolution1D(dim=8).model,
            cov=lambda l: l,
            name="y"
        )
    ]

    # Data
    data = cuqi.testproblem.Deconvolution1D(dim=8).data

    # Create a joint distribution
    J = cuqi.distribution.JointDistribution(densities)

    # Allow reduce
    J._allow_reduce = True

    # Check if we get the expected result when conditioning.
    assert isinstance(J(y=data), cuqi.distribution.JointDistribution)
    assert isinstance(J(y=data, z=1, d=1, l=1), cuqi.distribution.Posterior)
    assert isinstance(J(y=data, z=1, d=1, x=np.zeros(8)), cuqi.distribution.Posterior)
    assert isinstance(J(y=data, z=1, x=np.zeros(8), l=1), cuqi.distribution.Posterior)
    assert isinstance(J(y=data, d=1, x=np.zeros(8), l=1), cuqi.distribution.JointDistribution) # 2 likelihoods







