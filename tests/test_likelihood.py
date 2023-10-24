import cuqi
import pytest
import numpy as np
import scipy.sparse as sps
from pytest import approx

def test_likelihood_log_and_grad():
    #Create likelihood
    model, data, probInfo = cuqi.testproblem.Deconvolution1D().get_components()
    likelihood = cuqi.distribution.Gaussian(model,1).to_likelihood(data)

    # Tests log and gradient calls do not cause errors
    likelihood.logd(probInfo.exactSolution)
    likelihood.gradient(probInfo.exactSolution)

def test_likelihood_attributes():
    model, data, _ = cuqi.testproblem.Poisson1D(dim=128, field_type="Step").get_components()
    likelihood = cuqi.distribution.Gaussian(model,1).to_likelihood(data)

    # dim of domain
    assert likelihood.dim == model.domain_dim

    # Geometry of domain
    assert likelihood.geometry == model.domain_geometry

    # Shape of domain
    assert likelihood.par_shape == model.domain_geometry.par_shape
    assert likelihood.fun_shape == model.domain_geometry.fun_shape

    # Model extraction
    assert likelihood.model == model

@pytest.mark.parametrize("dist",[
                        # ------------ Scalar --------------
                        cuqi.distribution.Gaussian(np.zeros(128), sqrtcov=np.pi),
                        cuqi.distribution.Gaussian(np.zeros(128), cov=np.pi**2),
                        cuqi.distribution.Gaussian(np.zeros(128), prec=1/np.pi**2),
                        cuqi.distribution.Gaussian(np.zeros(128), sqrtprec=1/np.pi),
                        # ------------ VECTOR --------------
                        cuqi.distribution.Gaussian(np.zeros(128), sqrtcov=np.pi*np.ones(128)),
                        cuqi.distribution.Gaussian(np.zeros(128), cov=(np.pi**2)*np.ones(128)),
                        cuqi.distribution.Gaussian(np.zeros(128), prec=1/(np.pi**2)*np.ones(128)),
                        cuqi.distribution.Gaussian(np.zeros(128), sqrtprec=1/np.pi*np.ones(128)),
                        # ------------ Diagonal matrix --------------
                        cuqi.distribution.Gaussian(np.zeros(128), sqrtcov=np.pi*np.eye(128)),
                        cuqi.distribution.Gaussian(np.zeros(128), cov=(np.pi**2)*np.eye(128)),
                        cuqi.distribution.Gaussian(np.zeros(128), prec=1/(np.pi**2)*np.eye(128)),
                        cuqi.distribution.Gaussian(np.zeros(128), sqrtprec=1/np.pi*np.eye(128)),
                        # ------------ Sparse diagonal matrix --------------
                        cuqi.distribution.Gaussian(np.zeros(128), sqrtcov=np.pi*sps.eye(128)),
                        cuqi.distribution.Gaussian(np.zeros(128), cov=(np.pi**2)*sps.eye(128)),
                        cuqi.distribution.Gaussian(np.zeros(128), prec=1/(np.pi**2)*sps.eye(128)),
                        cuqi.distribution.Gaussian(np.zeros(128), sqrtprec=1/np.pi*sps.eye(128)),
                        ])
def test_likelihood_Gaussian_log_IID(dist):
    model, data, probInfo = cuqi.testproblem.Deconvolution1D(dim=128).get_components()
    dist.mean = model
    likelihood = dist.to_likelihood(probInfo.exactData) #We use exact data to get same logpdf every time
    assert likelihood.logd(probInfo.exactSolution) == approx(-264.14955763892135)

def test_likelihood_UserDefined():
    # CUQI likelihood
    model, data, probInfo = cuqi.testproblem.Deconvolution1D().get_components()
    L = cuqi.distribution.Gaussian(model, 1).to_likelihood(data)

    # Create user defined likelihood
    likelihood = cuqi.likelihood.UserDefinedLikelihood(dim=L.dim, logpdf_func=L.logd, gradient_func=L.gradient, geometry=L.geometry)
    
    # log
    assert likelihood.logd(probInfo.exactSolution) == L.logd(probInfo.exactSolution)

    # gradient
    assert np.allclose(likelihood.gradient(probInfo.exactSolution), L.gradient(probInfo.exactSolution))

    # dim
    assert likelihood.dim == model.domain_dim

    # geometry
    assert likelihood.geometry == model.domain_geometry

@pytest.mark.parametrize("dist",[
    cuqi.distribution.Gaussian(geometry=2),
    cuqi.distribution.Gaussian(lambda x: x, lambda s: s, geometry=2),
    cuqi.distribution.Gaussian(cuqi.model.Model(lambda x: x, 2, 2), lambda s:s, geometry=2)
])
@pytest.mark.parametrize("mean, cov, data",[
    (np.zeros(2), np.eye(2), np.ones(2)),
    (np.ones(2), 0.1*np.eye(2), np.random.rand(2)),
])
def test_likelihood_conditioning(dist, mean, cov, data):
    """ Test conditioning on parameters of likelihood for Gaussian """

    # Create likelihood
    likelihood = dist.to_likelihood(data)

    # Get parameter names
    param_names = likelihood.get_parameter_names()

    # Create dict for mean and cov (assumes mean is first parameter, which is not always true)
    mean_dict = {param_names[0]: mean}
    cov_dict = {param_names[1]: cov}

    # Full param dict
    param_dict = {**mean_dict, **cov_dict}

    # Evaluate logp from full params
    log_val = likelihood.logd(**param_dict)

    # Now compare log when conditioning on 3 cases: mean, cov, both
    assert likelihood(**mean_dict).logd(**cov_dict) == log_val
    assert likelihood(**cov_dict).logd(**mean_dict) == log_val
    assert likelihood(**param_dict).logd() == log_val # This becomes EvaluatedDensity. Throws warning on the stack size (since we pass name to EvaluatedDensity)


@pytest.mark.parametrize("y",
                         [cuqi.distribution.Gaussian(np.zeros(6), np.eye(6)),
                          cuqi.distribution.Lognormal(np.zeros(6), 4)])
@pytest.mark.parametrize("x_i", [np.array([0.1, 0.3, 6, 12, 1, 2]),
                                 np.array([0.1, 0.3, 0.5, 6, 3, 1])])
def test_enable_FD_gradient(y, x_i):
    """ Test that the likelihood exact gradient and FD gradient are close."""

    # Create a model
    model = cuqi.testproblem.Deconvolution1D(dim=6).model

    # Create likelihood
    y.mean = model
    data = y(x_i).sample()
    likelihood = y(y=data)

    # Compute exact gradient
    g_exact = likelihood.gradient(x_i)

    # Compute FD gradient
    likelihood.enable_FD(1e-7)
    g_FD = likelihood.gradient(x_i)

    # Assert that the exact and FD gradient are close,
    # but not exactly equal (since we use a different method)
    assert np.allclose(g_exact, g_FD) and np.all(g_exact != g_FD)
