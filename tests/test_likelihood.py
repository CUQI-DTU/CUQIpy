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
    likelihood.log(probInfo.exactSolution)
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
                        #cuqi.distribution.GaussianPrec(np.zeros(128), prec=np.eye(128)),
                        cuqi.distribution.GaussianSqrtPrec(np.zeros(128), sqrtprec=1/np.pi*np.eye(128)),
                        # ------------ Sparse diagonal matrix --------------
                        #cuqi.distribution.Gaussian(np.zeros(128), std=np.ones(128), corrmat=sps.eye(128)),
                        cuqi.distribution.GaussianCov(np.zeros(128), cov=(np.pi**2)*sps.eye(128)),
                        #cuqi.distribution.GaussianPrec(np.zeros(128), prec=np.eye(128)),
                        cuqi.distribution.GaussianSqrtPrec(np.zeros(128), sqrtprec=1/np.pi*sps.eye(128)),
                        ])
def test_likelihood_Gaussian_log_IID(dist):
    model, data, probInfo = cuqi.testproblem.Deconvolution1D.get_components(dim=128)
    dist.mean = model
    likelihood = dist.to_likelihood(probInfo.exactData) #We use exact data to get same logpdf every time
    assert likelihood.log(probInfo.exactSolution) == approx(-264.14955763892135)

def test_likelihood_UserDefined():
    # CUQI likelihood
    model, data, probInfo = cuqi.testproblem.Deconvolution1D.get_components()
    L = cuqi.distribution.GaussianCov(model, 1).to_likelihood(data)

    # Create user defined likelihood
    likelihood = cuqi.likelihood.UserDefinedLikelihood(dim=L.dim, logpdf_func=L.log, gradient_func=L.gradient, geometry=L.geometry)
    
    # log
    assert likelihood.log(probInfo.exactSolution) == L.log(probInfo.exactSolution)

    # gradient
    assert np.allclose(likelihood.gradient(probInfo.exactSolution), L.gradient(probInfo.exactSolution))

    # dim
    assert likelihood.dim == model.domain_dim

    # geometry
    assert likelihood.geometry == model.domain_geometry