import cuqi
import pytest
import numpy as np

def test_BayesianProblem_init():
    """ Test BayesianProblem initialization options for likelihood + prior """
    
    f = lambda x: np.hstack((x+5, x-5))
    A = cuqi.model.Model(f, 10, 5)
    x = cuqi.distribution.Gaussian(np.zeros(A.domain_dim), 1)
    y = cuqi.distribution.Gaussian(A, 1)
    y_obs = y(x.sample()).sample() # Sample prior, then sample data distribution

    # Posterior manually
    P = cuqi.distribution.Posterior(y.to_likelihood(y_obs), x)

    # Various ways to define BP
    BP1 = cuqi.problem.BayesianProblem(x, y).set_data(y=y_obs)    # Set data method
    BP2 = cuqi.problem.BayesianProblem(x, y, y=y_obs)             # data as keyword
    BP3 = cuqi.problem.BayesianProblem(y.to_likelihood(y_obs), x) # likelihood, prior

    # Check that all are the same
    _assert_same_BP_posterior(BP1, P)    
    _assert_same_BP_posterior(BP2, P)
    _assert_same_BP_posterior(BP3, P)

def _assert_same_BP_posterior(BP: cuqi.problem.BayesianProblem, P: cuqi.distribution.Posterior):
    assert np.allclose(BP.posterior.prior.mean, P.prior.mean)
    assert np.allclose(BP.posterior.prior.cov, P.prior.cov)
    assert BP.posterior.likelihood.distribution.mean == P.likelihood.distribution.mean # model
    assert np.allclose(BP.posterior.likelihood.distribution.cov, P.likelihood.distribution.cov)

def test_BayesianProblem_wrong_names():
    """ Test that wrong names raise errors """

    z = cuqi.distribution.Gaussian(np.zeros(10), 1) # p(z)
    y = cuqi.distribution.Gaussian(lambda x: x, 1)  # p(y|x)

    with pytest.raises(ValueError, match=r"Missing prior for"):
        cuqi.problem.BayesianProblem(z, y)

def test_BayesianProblem_hier_errors():
    """ Test that hierarchical Bayes models raise various errors (for now) """

    d = cuqi.distribution.Gamma(1, 1)
    x = cuqi.distribution.Gaussian(np.zeros(10), lambda d: d)
    y = cuqi.distribution.Gaussian(lambda x: x, 1)

    BP = cuqi.problem.BayesianProblem(y, x, d)

    with pytest.raises(ValueError, match=r"Unable to extract posterior"):
        BP.posterior

    with pytest.raises(ValueError, match=r"Unable to extract likelihood"):
        BP.likelihood

    with pytest.raises(ValueError, match=r"Unable to extract prior"):
        BP.prior

    with pytest.raises(ValueError, match=r"Unable to extract prior"):
        BP.sample_prior(10)

    with pytest.raises(NotImplementedError, match=r"Unable to determine sampling strategy"):
        BP.UQ()

    with pytest.raises(NotImplementedError, match=r"Unable to determine sampling strategy"):
        BP.sample_posterior(10)

    with pytest.raises(ValueError, match=r"Unable to extract prior"): # Should rly be posterior
        BP.MAP()

    with pytest.raises(ValueError, match=r"Unable to extract likelihood"):
        BP.ML()

    with pytest.raises(ValueError, match=r"Unable to extract likelihood"):
        BP.data

    with pytest.raises(ValueError, match=r"Unable to extract likelihood"):
        BP.model

def test_cannot_set_data():

    x = cuqi.distribution.Gaussian(0, 1)
    y = cuqi.distribution.Gaussian(lambda x: x, 1)

    BP = cuqi.problem.BayesianProblem(y, x)

    # Joint distribution does not care if we set the wrong data name (nothing happens)
    # We add a check here to catch this in the future when its fixed.
    BP.set_data(z=1) # z is not a valid name, should raise ValueError in future

    # Set data
    BP.set_data(y=1)

    # Try to set data again
    with pytest.raises(ValueError, match=r"data is already set"):
        BP.set_data(y=1)

def test_BayesianProblem_geometry_consistency():

    domain_geometry = cuqi.geometry.StepExpansion(np.arange(5), 3)
    range_geometry = cuqi.geometry.Continuous1D(10)

    f = lambda x: np.hstack((x+5, x-5))
    A = cuqi.model.Model(f, range_geometry, domain_geometry)
    x = cuqi.distribution.Gaussian(np.zeros(A.domain_dim), 1)
    y = cuqi.distribution.Gaussian(A, 1)
    y_obs = y(x.sample()).sample() # Sample prior, then sample data distribution

    # BayesianProblem
    BP = cuqi.problem.BayesianProblem(x, y).set_data(y=y_obs)

    # Check that we do not change the user provided geometries (We may want to change these in BP!)
    # This is what the now removed _check_geometry_consistency did (but it did not do it for all cases)
    # The tasks should be delegated to Posterior or JointDistribution
    assert BP.posterior.prior.geometry == x.geometry # prior geometry is not changed
    assert BP.posterior.likelihood.distribution.geometry == y.geometry # likelihood geometry is not changed
    assert BP.data.geometry == y_obs.geometry # data geometry is not changed
    assert BP.model == A # model geometry is not changed
    
    # Check that BayesianProblem correctly infers the geometry of the posterior (from model)
    assert BP.posterior.geometry == domain_geometry
    assert BP.sample_posterior(10).geometry == domain_geometry
    #assert BP.sample_prior(10).geometry == domain_geometry # This fails because prior is sampled directly. TODO (should be domain_geometry)
    assert BP.MAP().geometry == domain_geometry
    assert BP.ML().geometry == domain_geometry


def test_passing_burnin_to_UQ_method(capfd):
    """ Test that passing burnin to UQ method works correctly"""
    # Create simple forward model
    fwd_model = cuqi.model.Model(lambda x: x, 1, 1)

    # Create prior and data distributions
    x = cuqi.distribution.Gaussian(0, 1)
    y = cuqi.distribution.Gaussian(fwd_model(x), 1)

    # Create BayesianProblem and set data
    BP = cuqi.problem.BayesianProblem(x, y)
    BP.set_data(y=1)

    # Apply UQ method with burnin
    samples = BP.UQ(Ns=10, Nb=5) 

    # Read output
    out, err = capfd.readouterr()

    # Check that correct burnin is used
    assert "50%" in out
    assert "Sample 15 / 15" in out
