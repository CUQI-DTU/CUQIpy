import cuqi
import pytest
import numpy as np
import scipy.sparse as sps
from pytest import approx

def test_likelihood_log_and_grad():
    model, data, probInfo = cuqi.testproblem.Deconvolution.get_components()
    likelihood = cuqi.distribution.GaussianCov(model,1).to_likelihood(data)
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
                        cuqi.distribution.Gaussian(np.zeros(128), std=1),
                        cuqi.distribution.GaussianCov(np.zeros(128), cov=1),
                        #cuqi.distribution.GaussianPrec(np.zeros(128), prec=1),
                        cuqi.distribution.GaussianSqrtPrec(np.zeros(128), sqrtprec=1),
                        # ------------ VECTOR --------------
                        cuqi.distribution.Gaussian(np.zeros(128), std=np.ones(128)),
                        cuqi.distribution.GaussianCov(np.zeros(128), cov=np.ones(128)),
                        #cuqi.distribution.GaussianPrec(np.zeros(128), prec=np.ones(128)),
                        cuqi.distribution.GaussianSqrtPrec(np.zeros(128), sqrtprec=np.ones(128)),
                        # ------------ Diagonal matrix --------------
                        cuqi.distribution.Gaussian(np.zeros(128), std=np.ones(128), corrmat=np.eye(128)),
                        cuqi.distribution.GaussianCov(np.zeros(128), cov=np.eye(128)),
                        #cuqi.distribution.GaussianPrec(np.zeros(128), prec=np.eye(128)),
                        cuqi.distribution.GaussianSqrtPrec(np.zeros(128), sqrtprec=np.eye(128)),
                        # ------------ Sparse diagonal matrix --------------
                        #cuqi.distribution.Gaussian(np.zeros(128), std=np.ones(128), corrmat=sps.eye(128)),
                        cuqi.distribution.GaussianCov(np.zeros(128), cov=sps.eye(128)),
                        #cuqi.distribution.GaussianPrec(np.zeros(128), prec=np.eye(128)),
                        cuqi.distribution.GaussianSqrtPrec(np.zeros(128), sqrtprec=sps.eye(128)),
                        ])
def test_likelihood_Gaussian_log_IID(dist):
    model, data, probInfo = cuqi.testproblem.Deconvolution.get_components(dim=128)
    dist.mean = model
    likelihood = dist.to_likelihood(probInfo.exactData) #We use exact data to get same logpdf every time
    assert likelihood.log(probInfo.exactSolution) == approx(-117.6241322501981)

def test_likelihood_UserDefined():
    # CUQI likelihood
    model, data, probInfo = cuqi.testproblem.Deconvolution.get_components()
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