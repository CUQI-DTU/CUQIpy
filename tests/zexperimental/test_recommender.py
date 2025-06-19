import cuqi
import pytest
import numpy as np
import inspect
from numbers import Number


def test_find_valid_samplers_linearGaussianGaussian():
    target = cuqi.testproblem.Deconvolution1D(dim=2).posterior

    valid_samplers = cuqi.experimental.SamplerRecommender(target).valid_samplers()

    assert(set(valid_samplers) == set(['CWMH', 'LinearRTO', 'MALA', 'MH', 'NUTS', 'PCN', 'ULA']))

def test_find_valid_samplers_nonlinearGaussianGaussian():
    target = cuqi.testproblem.Poisson1D(dim=2).posterior

    valid_samplers = cuqi.experimental.SamplerRecommender(target).valid_samplers()

    print(set(valid_samplers) == set(['CWMH', 'MH', 'PCN']))

def test_find_valid_samplers_conjugate_valid():
    """ Test that conjugate sampler is valid for Gaussian-Gamma conjugate pair when parameter is defined as the precision."""
    x = cuqi.distribution.Gamma(1,1)
    y = cuqi.distribution.Gaussian(np.zeros(2), cov=lambda x : 1/x) # Valid on precision only, e.g. cov=lambda x : 1/x
    target = cuqi.distribution.JointDistribution(y, x)(y = 1)

    valid_samplers = cuqi.experimental.SamplerRecommender(target).valid_samplers()

    assert(set(valid_samplers) == set(['CWMH', 'Conjugate', 'MH']))

def test_find_valid_samplers_conjugate_invalid():
    """ Test that conjugate sampler is invalid for Gaussian-Gamma conjugate pair when parameter is defined as the covariance."""
    x = cuqi.distribution.Gamma(1,1)
    y = cuqi.distribution.Gaussian(np.zeros(2), cov=lambda x : x) # Invalid if defined via covariance as cov=lambda x : x
    target = cuqi.distribution.JointDistribution(y, x)(y = 1)

    valid_samplers = cuqi.experimental.SamplerRecommender(target).valid_samplers()

    assert(set(valid_samplers) == set(['CWMH', 'MH']))

def test_find_valid_samplers_direct():
    target = cuqi.distribution.Gamma(1,1)

    valid_samplers = cuqi.experimental.SamplerRecommender(target).valid_samplers()

    assert(set(valid_samplers) == set(['CWMH', 'Direct', 'MH']))

def test_find_valid_samplers_implicit_posterior():
    A, y_obs, _ = cuqi.testproblem.Deconvolution1D(dim=2).get_components()

    x = cuqi.implicitprior.RegularizedGaussian(np.zeros(2), 1, constraint="nonnegativity")
    y = cuqi.distribution.Gaussian(A@x, 1)
    target =  cuqi.distribution.JointDistribution(y, x)(y = y_obs)

    valid_samplers = cuqi.experimental.SamplerRecommender(target).valid_samplers()

    assert(set(valid_samplers) == set(['RegularizedLinearRTO']))

def test_find_valid_samplers_implicit_prior():
    target = cuqi.implicitprior.RegularizedGaussian(np.zeros(2), 1, constraint="nonnegativity")

    valid_samplers = cuqi.experimental.SamplerRecommender(target).valid_samplers()

    assert(len(set(valid_samplers)) == 0)