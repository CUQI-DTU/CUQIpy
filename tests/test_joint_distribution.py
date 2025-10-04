import cuqi
import numpy as np
import pytest
from .test_model import MultipleInputTestModel

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

def test_joint_distribution_with_multiple_inputs_model_has_correct_parameter_names():
    """Test that the joint distribution based on model with multiple inputs has
    correct parameter names."""

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

    x_val = np.array([1, 2, 3])
    y_val = np.array([4, 5])
    z_val = np.array([6, 7, 8])

    posterior = cuqi.distribution.JointDistribution(
        likelihood,
        x_dist,
        y_dist,
        z_dist
    )

    # Ensure correct parameter names are returned for joint distribution with likelihood
    assert posterior.get_parameter_names() == ['x_dist', 'y_dist', 'z_dist']

    assert posterior(x_dist=x_val).get_parameter_names() == ['y_dist', 'z_dist']
    assert posterior(y_dist=y_val).get_parameter_names() == ['x_dist', 'z_dist']
    assert posterior(z_dist=z_val).get_parameter_names() == ['x_dist', 'y_dist']

    assert posterior(y_dist=y_val, z_dist=z_val).get_parameter_names() == ['x_dist']
    assert posterior(x_dist=x_val, z_dist=z_val).get_parameter_names() == ['y_dist']
    assert posterior(x_dist=x_val, y_dist=y_val).get_parameter_names() == ['z_dist']

    assert posterior(x_dist=x_val, y_dist=y_val, z_dist=z_val).get_parameter_names() == []

    joint_dist =  cuqi.distribution.JointDistribution(
        data_dist,
        x_dist,
        y_dist,
        z_dist
    )

    # Ensure correct parameter names are returned for joint distribution with data distribution
    assert joint_dist.get_parameter_names() == ['data_dist', 'x_dist', 'y_dist', 'z_dist']
    assert joint_dist(x_dist=x_val).get_parameter_names() == ['data_dist', 'y_dist', 'z_dist']
    assert joint_dist(y_dist=y_val).get_parameter_names() == ['data_dist', 'x_dist', 'z_dist']
    assert joint_dist(z_dist=z_val).get_parameter_names() == ['data_dist', 'x_dist', 'y_dist']
    assert joint_dist(data_dist=np.array([2,2,3])).get_parameter_names() == ['x_dist', 'y_dist', 'z_dist']

    assert joint_dist(x_dist=x_val, data_dist=np.array([2,2,3])).get_parameter_names() == ['y_dist', 'z_dist']
    assert joint_dist(y_dist=y_val, data_dist=np.array([2,2,3])).get_parameter_names() == ['x_dist', 'z_dist']
    assert joint_dist(z_dist=z_val, data_dist=np.array([2,2,3])).get_parameter_names() == ['x_dist', 'y_dist']
    assert joint_dist(x_dist=x_val, y_dist=y_val).get_parameter_names() == ['data_dist', 'z_dist']
    assert joint_dist(x_dist=x_val, z_dist=z_val).get_parameter_names() == ['data_dist', 'y_dist']
    assert joint_dist(y_dist=y_val, z_dist=z_val).get_parameter_names() == ['data_dist', 'x_dist']

    assert joint_dist(x_dist=x_val, y_dist=y_val, z_dist=z_val).get_parameter_names() == ['data_dist']
    assert joint_dist(x_dist=x_val, y_dist=y_val, data_dist=np.array([2,2,3])).get_parameter_names() == ['z_dist']
    assert joint_dist(x_dist=x_val, z_dist=z_val, data_dist=np.array([2,2,3])).get_parameter_names() == ['y_dist']
    assert joint_dist(y_dist=y_val, z_dist=z_val, data_dist=np.array([2,2,3])).get_parameter_names() == ['x_dist']

    # Ensure correct parameter names are returned for underlying likelihood
    assert joint_dist(data_dist=np.array([2,2,3]))._likelihoods[0].get_parameter_names() == ['x_dist', 'y_dist', 'z_dist']

    assert joint_dist(x_dist=x_val, data_dist=np.array([2,2,3]))._likelihoods[0].get_parameter_names() == ['y_dist', 'z_dist']
    assert joint_dist(y_dist=y_val, data_dist=np.array([2,2,3]))._likelihoods[0].get_parameter_names() == ['x_dist', 'z_dist']
    assert joint_dist(z_dist=z_val, data_dist=np.array([2,2,3]))._likelihoods[0].get_parameter_names() == ['x_dist', 'y_dist']

    assert joint_dist(x_dist=x_val, y_dist=y_val, data_dist=np.array([2,2,3])).likelihood.get_parameter_names() == ['z_dist']
    assert joint_dist(x_dist=x_val, z_dist=z_val, data_dist=np.array([2,2,3])).likelihood.get_parameter_names() == ['y_dist']
    assert joint_dist(y_dist=y_val, z_dist=z_val, data_dist=np.array([2,2,3])).likelihood.get_parameter_names() == ['x_dist']


def test_FD_enabled_is_set_correctly():
    """ Test that FD_enabled property is set correctly in JointDistribution """

    # Create a joint distribution with two distributions
    d1 = cuqi.distribution.Normal(0, 1, name="x")
    d2 = cuqi.distribution.Gamma(lambda x: x**2, 1, name="y")
    J = cuqi.distribution.JointDistribution(d1, d2)

    # Initially FD should be disabled for both
    assert J.FD_enabled == {"x": False, "y": False}

    # Enable FD for x
    J.enable_FD(epsilon={"x": 1e-6, "y": None})
    assert J.FD_enabled == {"x": True, "y": False}
    assert J.FD_epsilon == {"x": 1e-6, "y": None}

    # Enable FD for y as well
    J.enable_FD(epsilon={"x": 1e-6, "y": 1e-5})
    assert J.FD_enabled == {"x": True, "y": True}
    assert J.FD_epsilon == {"x": 1e-6, "y": 1e-5}

    # Disable FD for x
    J.enable_FD(epsilon={"x": None, "y": 1e-5})
    assert J.FD_enabled == {"x": False, "y": True}
    assert J.FD_epsilon == {"x": None, "y": 1e-5}

    # Disable FD for all
    J.disable_FD()
    assert J.FD_enabled == {"x": False, "y": False}
    assert J.FD_epsilon == {"x": None, "y": None}

    # Enable FD and reduce to single density
    J.enable_FD() # Enable FD for all
    J_given_x = J(x=0)
    J_given_y = J(y=1)

    # Check types and FD_enabled status of J_given_x
    assert isinstance(J_given_x, cuqi.distribution.Gamma)
    assert not J_given_x.FD_enabled # intentionally disabled for single remaining
                                    # distribution
    assert J_given_x.FD_epsilon == None

    # Check types and FD_enabled status of J_given_y
    assert isinstance(J_given_y, cuqi.distribution.Posterior)
    assert J_given_y.FD_enabled
    assert J_given_y.FD_epsilon == 1e-8 # Default epsilon for remaining density

    # Catch error if epsilon keys do not match parameter names
    with pytest.raises(ValueError, match=r"Keys of FD_epsilon must match"):
        J.enable_FD(epsilon={"x": 1e-6}) # Missing "y" key

def test_FD_enabled_is_set_correctly_for_stacked_joint_distribution():
    """ Test that FD_enabled property is set correctly in JointDistribution """

    # Create a joint distribution with two distributions
    x = cuqi.distribution.Normal(0, 1, name="x")
    y = cuqi.distribution.Uniform(1, 2, name="y")
    J = cuqi.distribution._StackedJointDistribution(x, y)
    J.enable_FD(epsilon={"x": 1e-6, "y": None})

    assert J.FD_enabled == {"x": True, "y": False}
    assert J.FD_epsilon == {"x": 1e-6, "y": None}

    # Reduce to single density (substitute y)
    J_given_y = J(y=1.5)
    assert isinstance(J_given_y, cuqi.distribution.Normal)
    assert J_given_y.FD_enabled == False # Intentionally disabled for
                                         # single remaining
                                         # distribution
    assert J_given_y.FD_epsilon is None

    # Reduce to single density (substitute x)
    J_given_x = J(x=0)
    assert isinstance(J_given_x, cuqi.distribution.Uniform)
    assert J_given_x.FD_enabled == False
    assert J_given_x.FD_epsilon is None



@pytest.mark.parametrize(
    "densities,kwargs,fd_epsilon,expected_type,expected_fd_enabled",
    [
        # Case 0: Single Distribution, FD enabled
        (
            [cuqi.distribution.Normal(np.zeros(3), 1, name="x")],
            {},
            {"x": 1e-5},
            cuqi.distribution.Normal,
            False,  # Intentionally disabled for single remaining distribution
        ),
        # Case 1: Single Distribution, FD disabled
        (
            [cuqi.distribution.Normal(np.zeros(3), 1, name="x")],
            {},
            {"x": None},
            cuqi.distribution.Normal,
            False,
        ),
        # Case 2: Distribution + Data distribution, substitute y
        (
            [
                cuqi.distribution.Normal(np.zeros(3), 1, name="x"),
                cuqi.distribution.Gaussian(lambda x: x**2, np.ones(3), name="y"),
            ],
            {"y": np.ones(3)},
            {"x": 1e-6, "y": 1e-7},
            cuqi.distribution.Posterior,
            True,
        ),
        # Case 3: Distribution + data distribution, substitute x
        (
            [
                cuqi.distribution.Normal(np.zeros(3), 1, name="x"),
                cuqi.distribution.Gaussian(lambda x: x**2, np.ones(3), name="y"),
            ],
            {"x": np.ones(3)},
            {"x": 1e-5, "y": 1e-6},
            cuqi.distribution.Distribution,
            False,  # Intentionally disabled for single remaining distribution
        ),
        # Case 4: Multiple data distributions + prior (MultipleLikelihoodPosterior)
        (
            [
                cuqi.distribution.Normal(np.zeros(3), 1, name="x"),
                cuqi.distribution.Gaussian(lambda x: x, np.ones(3), name="y1"),
                cuqi.distribution.Gaussian(lambda x: x + 1, np.ones(3), name="y2"),
            ],
            {"y1": np.ones(3), "y2": np.ones(3)},
            {"x": 1e-5, "y1": 1e-6, "y2": 1e-7},
            cuqi.distribution.MultipleLikelihoodPosterior,
            {"x": True},
        ),
        # Case 5: Distribution, substitute x
        (
            [cuqi.distribution.Normal(np.zeros(3), 1, name="x")],
            {"x": np.ones(3)},
            {"x": 1e-8},
            cuqi.distribution.JointDistribution,
            {},
        ),
    ],
)
def test_fd_enabled_of_joint_distribution_after_substitution_is_correct(
    densities, kwargs, fd_epsilon, expected_type, expected_fd_enabled
):
    """ Test that FD_enabled and FD_epsilon properties are set correctly in JointDistribution even after substitution."""
    joint = cuqi.distribution.JointDistribution(*densities)
    joint.enable_FD(epsilon=fd_epsilon)

    # Assert FD_epsilon is set correctly
    assert joint.FD_epsilon == fd_epsilon

    # Substitute parameters (if any), which reduces the joint distribution
    reduced = joint(**kwargs)

    # Assert the type and FD_enabled status of the reduced distribution
    assert isinstance(reduced, expected_type)
    assert reduced.FD_enabled == expected_fd_enabled

    # Assert FD_epsilon is set correctly in the reduced distribution
    if expected_fd_enabled is not False:
        fd_epsilon_reduced = {
            k: v for k, v in fd_epsilon.items() if k not in kwargs.keys()
        }
        if len(fd_epsilon_reduced) == 1 and not isinstance(
            reduced, cuqi.distribution.MultipleLikelihoodPosterior
        ):
            # Single value instead of dict in this case
            fd_epsilon_reduced = list(fd_epsilon_reduced.values())[0]
        assert reduced.FD_epsilon == fd_epsilon_reduced
