import cuqi
import pytest
import numpy as np
import scipy.sparse as sps
from pytest import approx

def test_likelihood_log_and_grad():
    #Create likelihood
    model, data, probInfo = cuqi.testproblem.Deconvolution1D.get_components()
    likelihood = cuqi.distribution.GaussianCov(model,1).to_likelihood(data)

    # Tests log and gradient calls do not cause errors
    likelihood.logd(probInfo.exactSolution)
    likelihood.gradient(probInfo.exactSolution)

def test_likelihood_attributes():
    model, data, _ = cuqi.testproblem.Poisson_1D.get_components(dim=128, field_type="Step")
    likelihood = cuqi.distribution.GaussianCov(model,1).to_likelihood(data)

    # dim of domain
    assert likelihood.dim == model.domain_dim

    # Geometry of domain
    assert likelihood.geometry == model.domain_geometry

    # Shape of domain
    assert likelihood.shape == model.domain_geometry.shape

    # Model extraction
    assert likelihood.model == model

@pytest.mark.parametrize("dist",[
                        # ------------ Scalar --------------
                        cuqi.distribution.Gaussian(np.zeros(128), std=np.pi),
                        cuqi.distribution.GaussianCov(np.zeros(128), cov=np.pi**2),
                        #cuqi.distribution.GaussianPrec(np.zeros(128), prec=1),
                        cuqi.distribution.GaussianSqrtPrec(np.zeros(128), sqrtprec=1/np.pi),
                        # ------------ VECTOR --------------
                        cuqi.distribution.Gaussian(np.zeros(128), std=np.pi*np.ones(128)),
                        cuqi.distribution.GaussianCov(np.zeros(128), cov=(np.pi**2)*np.ones(128)),
                        #cuqi.distribution.GaussianPrec(np.zeros(128), prec=np.ones(128)),
                        cuqi.distribution.GaussianSqrtPrec(np.zeros(128), sqrtprec=1/np.pi*np.ones(128)),
                        # ------------ Diagonal matrix --------------
                        cuqi.distribution.Gaussian(np.zeros(128), std=np.pi*np.ones(128), corrmat=np.eye(128)),
                        cuqi.distribution.GaussianCov(np.zeros(128), cov=(np.pi**2)*np.eye(128)),
                        cuqi.distribution.GaussianPrec(np.zeros(128), prec=1/(np.pi**2)*np.eye(128)),
                        cuqi.distribution.GaussianSqrtPrec(np.zeros(128), sqrtprec=1/np.pi*np.eye(128)),
                        # ------------ Sparse diagonal matrix --------------
                        #cuqi.distribution.Gaussian(np.zeros(128), std=np.ones(128), corrmat=sps.eye(128)),
                        cuqi.distribution.GaussianCov(np.zeros(128), cov=(np.pi**2)*sps.eye(128)),
                        cuqi.distribution.GaussianPrec(np.zeros(128), prec=1/(np.pi**2)*sps.eye(128)),
                        cuqi.distribution.GaussianSqrtPrec(np.zeros(128), sqrtprec=1/np.pi*sps.eye(128)),
                        ])
def test_likelihood_Gaussian_log_IID(dist):
    model, data, probInfo = cuqi.testproblem.Deconvolution1D.get_components(dim=128)
    dist.mean = model
    likelihood = dist.to_likelihood(probInfo.exactData) #We use exact data to get same logpdf every time
    assert likelihood.logd(probInfo.exactSolution) == approx(-264.14955763892135)

def test_likelihood_UserDefined():
    # CUQI likelihood
    model, data, probInfo = cuqi.testproblem.Deconvolution1D.get_components()
    L = cuqi.distribution.GaussianCov(model, 1).to_likelihood(data)

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
    cuqi.distribution.GaussianCov(),
    cuqi.distribution.GaussianCov(lambda x: x, lambda s: s),
    cuqi.distribution.GaussianCov(cuqi.model.Model(lambda x: x, 2, 2), lambda s:s)
])
@pytest.mark.parametrize("mean, cov, data",[
    (np.zeros(2), np.eye(2), np.ones(2)),
    (np.zeros(3), np.eye(3), np.random.rand(3)),
])
def test_likelihood_conditioning(dist, mean, cov, data):
    """ Test conditioning on parameters of likelihood for GaussianCov """

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
    assert likelihood(**param_dict).logd() == log_val
