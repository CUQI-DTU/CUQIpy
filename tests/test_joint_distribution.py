import cuqi
import numpy as np
import pytest

def test_joint_dist_dim_geometry():
    """ Test the dimension and geometry properties of a joint distribution """

    model, data, _ = cuqi.testproblem.Poisson1D().get_components() # Model is m times n, m != n.

    # Bayesian model
    d = cuqi.distribution.Gamma(1, 1e-4)
    x = cuqi.distribution.Gaussian(np.zeros(model.domain_dim), lambda d: d)
    y = cuqi.distribution.Gaussian(model, 1)

    # Joint distribution
    J = cuqi.distribution.JointDistribution(d, x, y)

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
        cuqi.distribution.Gaussian(np.zeros(8), lambda d: d, name="x"),
        cuqi.distribution.Gaussian(
            mean=cuqi.testproblem.Deconvolution1D(dim=8).model,
            cov=lambda l: l,
            name="y"
        )
    ],
    [
        cuqi.distribution.Normal(0, 1e-2, name="z"),
        cuqi.distribution.Gamma(1, lambda z: abs(z), name="d"),
        cuqi.distribution.Gamma(lambda z: z, 1e-2, name="l"),
        cuqi.distribution.Gaussian(np.zeros(8), lambda d: d, name="x"),
        cuqi.distribution.Gaussian(
            mean=cuqi.testproblem.Deconvolution1D(dim=8).model,
            cov=lambda l: l,
            name="y"
        )
    ],
])
def test_joint_dist_logd(densities):
    """ Tests the log density function of the Joint distribution """

    # Create a joint distribution
    J = cuqi.distribution.JointDistribution(*densities)

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

@pytest.mark.parametrize("densities", [
    [
        cuqi.distribution.Gamma(1, 1e-4, name="x"),
        cuqi.distribution.Gaussian(0, lambda x:x, name="z"),
        cuqi.distribution.Normal(0, lambda x:x, name="y")
    ],
    [
        cuqi.distribution.Gamma(1, 1e-4, name="d"),
        cuqi.distribution.Gamma(1, 1e-2, name="l"),
        cuqi.distribution.Gaussian(np.zeros(8), lambda d: d, name="x"),
        cuqi.distribution.Gaussian(
            mean=cuqi.testproblem.Deconvolution1D(dim=8).model,
            cov=lambda l: l,
            name="y"
        )
    ],
    [
        cuqi.distribution.Normal(0, 1e-2, name="z"),
        cuqi.distribution.Gamma(1, lambda z: abs(z), name="d"),
        cuqi.distribution.Gamma(lambda z: z, 1e-2, name="l"),
        cuqi.distribution.Gaussian(np.zeros(8), lambda d: d, name="x"),
        cuqi.distribution.Gaussian(
            mean=cuqi.testproblem.Deconvolution1D(dim=8).model,
            cov=lambda l: l,
            name="y"
        )
    ],
])
def test_joint_dist_properties(densities):
    """ This tests various properties of the joint distribution """

    # Create a joint distribution
    J = cuqi.distribution.JointDistribution(*densities)

    # Now check various properties. First check the dimension
    assert J.dim == [density.dim for density in densities]

    # Check the geometry
    assert J.geometry == [density.geometry for density in densities]

    # Check the parameter names
    assert J.get_parameter_names() == [density.name for density in densities]

    # Check list of distributions
    assert J._distributions == densities

    # Check likelihoods
    assert J._likelihoods == []

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
    assert [dist.name for dist in P._distributions] == [density.name for density in densities[:-1]]

    # Check likelihoods by comparing the names
    assert [L.name for L in P._likelihoods] == [densities[-1].name]

def test_joint_dist_reduce():
    """ This tests the reduce hack for the joint distribution. """

    J, data = hierarchical_joint()   

    # Check if we get the expected result when conditioning.
    assert isinstance(J(y=data), cuqi.distribution.JointDistribution)
    assert isinstance(J(y=data, z=1, d=1, l=1), cuqi.distribution.Posterior)
    assert isinstance(J(y=data, z=1, d=1, x=np.zeros(8)), cuqi.distribution.Posterior)
    assert isinstance(J(y=data, z=1, x=np.zeros(8), l=1), cuqi.distribution.Posterior)
    assert isinstance(J(y=data, d=1, x=np.zeros(8), l=1), cuqi.distribution.JointDistribution) # 2 likelihoods

def test_stacked_joint_distribution():
    """ This tests the stacked joint distribution """

    J, data = hierarchical_joint()

    Js = J._as_stacked()

    # Check the dimension
    assert Js.dim == sum(J.dim)

    # Check the geometry
    assert Js.geometry == cuqi.geometry._DefaultGeometry1D(Js.dim)

    # Check you can evaluate the log density function with a single vector
    Js.logd(np.ones(Js.dim))

    # Check that you can condition on the stacked distribution
    # and again evaluate as a single vector
    Ps = Js(y=data)
    Ps.logd(np.ones(Ps.dim))

    # Even if you condition in multiple parameters
    Ps2 = Js(y=data, z=1, d=1)
    Ps2.logd(np.ones(Ps2.dim))

    # And even if you condition on all parameters
    Ps3 = Js(y=data, z=1, d=1, x=np.zeros(Js.get_density("x").dim), l=1)
    #Ps3.logd() # Should be allowed with no arguments

def hierarchical_joint(main_dim=8):
    """ This creates a deconvolution based hierarchical joint distribution.
    
    Parameters
    ----------
    main_dim : int
        The dimension of the main model.

    Returns
    -------
    JointDistribution, data
        
    """

    densities = [
        cuqi.distribution.Normal(0, 1e-2, name="z"),
        cuqi.distribution.Gamma(1, lambda z: abs(z), name="d"),
        cuqi.distribution.Gamma(lambda z: z, 1e-2, name="l"),
        cuqi.distribution.Gaussian(np.zeros(main_dim), lambda d: d, name="x"),
        cuqi.distribution.Gaussian(
            mean=cuqi.testproblem.Deconvolution1D(dim=main_dim).model,
            cov=lambda l: l,
            name="y"
        )
    ]

    data = cuqi.testproblem.Deconvolution1D(dim=main_dim).data

    J = cuqi.distribution.JointDistribution(*densities)

    return J, data

def test_extra_parameter_no_prior():
    A = np.random.randn(10,3)

    # Define distributions for Bayesian model
    y = cuqi.distribution.Normal(lambda x: A@x, np.ones(10))
    x = cuqi.distribution.Normal(np.zeros(3), lambda z,b:z+b)
    z = cuqi.distribution.Gamma(1, 1)

    # Joint distribution p(y,x,z)
    with pytest.raises(ValueError, match=r"Missing prior for \['b'\]"):
        cuqi.distribution.JointDistribution(y,x,z)


def test_JointDistribution_reduce_MultipleLikelihoodPosterior():
    """ This tests if the joint distribution can be reduced to MultipleLikelihoodPosterior """

    J, data = hierarchical_joint()

    # Posterior w.r.t z is with multiple likelihoods if all other parameters are fixed
    # for this J
    posterior_z = J(y=data, d=0.1, l=0.1, x=data)

    # Check we get the right class
    assert isinstance(posterior_z, cuqi.distribution.MultipleLikelihoodPosterior)

    # Check the dimension
    assert posterior_z.dim == 1

    # Check the "prior" in multiple likelihood posterior matches z
    assert posterior_z.prior.mean == J.get_density("z").mean
    assert posterior_z.prior.std == J.get_density("z").std

    # Check logd can be evaluated (tolerance is high since new data every time)
    assert np.allclose(posterior_z.logd(1), -5000, atol=10)

def test_MultipleLikelihoodPosterior_should_raise_if_two_densities():
    """ This tests if the MultipleLikelihoodPosterior raises if it has two densities. """

    # Define distributions for Bayesian model
    y = cuqi.distribution.Normal(lambda x: x, 1)
    x = cuqi.distribution.Normal(0, 1)

    # Joint distribution p(y,x)
    J = cuqi.distribution.JointDistribution(y,x)

    # Posterior
    P = J(y=1)

    # Check we get the right class
    assert isinstance(P, cuqi.distribution.Posterior)

    # Check that we cannot create MultipleLikelihoodPosterior
    with pytest.raises(ValueError, match=r"requires at least three densities"):
        cuqi.distribution.MultipleLikelihoodPosterior(y.to_likelihood(1), x)


def test_MultipleLikelihoodPosterior_should_raise_if_no_likelihood():
    """ This tests if the MultipleLikelihoodPosterior raises if no likelihood is given."""

    # Define distributions for Bayesian model
    y1 = cuqi.distribution.Normal(lambda x: x, 1)
    y2 = cuqi.distribution.Normal(lambda x: x+1, 1)
    x = cuqi.distribution.Normal(0, 1)
 
    # Check that we cannot create MultipleLikelihoodPosterior
    with pytest.raises(ValueError, match=r"must have a likelihood and prior"):
        cuqi.distribution.MultipleLikelihoodPosterior(y1, y2, x)

def test_MultipleLikelihoodPosterior_should_raise_if_names_do_not_match():
    """ This tests if the MultipleLikelihoodPosterior raises if the names do not match."""

    # Define distributions for Bayesian model
    y = cuqi.distribution.Normal(lambda x: x, 1)
    x = cuqi.distribution.Normal(0, 1)
    z = cuqi.distribution.Normal(0, 1)
 
    # Check that we cannot create MultipleLikelihoodPosterior
    with pytest.raises(ValueError, match=r"the same parameter name"):
        cuqi.distribution.MultipleLikelihoodPosterior(y.to_likelihood(1), x, z)

@pytest.mark.parametrize("joint, variables", [
    (
        cuqi.distribution.JointDistribution(
            cuqi.distribution.Normal(0, 1, name="x"),
            cuqi.distribution.Normal(0, 1, name="y")
        ),
        {
            "x": 0,
            "y": 0
        }
    ),
    (
        cuqi.distribution.JointDistribution(
            cuqi.distribution.Uniform(0, 10, name="d"),
            cuqi.distribution.Uniform(0, 5, name="s"),
            cuqi.distribution.Gaussian(np.zeros(8), lambda d: d, name="x"),
            cuqi.distribution.Gaussian(
                mean=cuqi.testproblem.Deconvolution1D(dim=8).model,
                cov=lambda s: s,
                name="y"
            )
        ),
        {
            "d": 5,
            "s": 2,
            "x": np.zeros(8),
            "y": cuqi.testproblem.Deconvolution1D(dim=8).data
        }
    ),
    (
        cuqi.distribution.JointDistribution(
            cuqi.distribution.Uniform(0, 10, name="d"),
            cuqi.distribution.Uniform(0, 5, name="s"),
            cuqi.distribution.Gaussian(np.zeros(8), lambda d: d, name="x"),
            cuqi.distribution.Gaussian(
                mean=cuqi.testproblem.Deconvolution1D(dim=8).model,
                cov=lambda s: s,
                name="y"
            )
        ),
        {
            "d": 11, # Out of bounds (all should return -Inf)
            "s": 2,
            "x": np.zeros(8),
            "y": cuqi.testproblem.Deconvolution1D(dim=8).data
        }
    )
    ]
)
def test_logd_consistency_when_conditioning(joint, variables):
    """ Test consistency of logd value when conditioning the joint distribution.
     
    This ensures we always return the correct value for logd even when reducing to single density.
    
    """

    # True value of logd by fully evaluating the joint distribution
    true_value = joint.logd(**variables)

    # Loop over all variables and evaluate the logd value
    # where all previously seen variables are conditioned
    # and not seen variables are given to logd.
    cond_vars = {}
    for key, value in variables.items():
        # Remaining variables are those that are not used for conditioning
        remaining_vars = {k: v for k, v in variables.items() if k not in cond_vars}
        
        # Condition the joint distribution
        cond_joint = joint(**cond_vars)

        # Evaluate the logd value of conditioned joint distribution
        # This may be a single density or a joint distribution since
        # joint can reduce to single density (such as Posterior)
        logd_value = cond_joint.logd(**remaining_vars)

        # Potential error message if this assert fails:
        msg = f"Evaluated: \n {joint} \n with variables {variables}.\nFailed at {key}={value} for \n {cond_joint} \n with variables {remaining_vars}."
        
        # Assert the logd value matches
        assert logd_value == pytest.approx(true_value, rel=1e-6), msg

        # Add current variable to the variables that need to be conditioned
        cond_vars[key] = value
