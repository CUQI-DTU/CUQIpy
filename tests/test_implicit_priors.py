import cuqi
import numpy as np
import pytest

def test_RegularizedGaussian_default_init():
    """ Test that the implicit regularized Gaussian requires at least 1 regularization argument """

    with pytest.raises(ValueError, match="Precisely one of "):
        x = cuqi.implicitprior.RegularizedGaussian(np.zeros(5), 1)

def test_RegularizedGaussian_guarding_statements():
    """ Test that we catch incorrect initialization of RegularizedGaussian """

    # More than 1 argument
    with pytest.raises(ValueError, match="Precisely one of "):
        cuqi.implicitprior.RegularizedGaussian(np.zeros(5), 1, proximal=lambda s,z: s, constraint="nonnegativity")

    # Proximal
    with pytest.raises(ValueError, match="Proximal needs to be callable"):
        cuqi.implicitprior.RegularizedGaussian(np.zeros(5), 1, proximal=1)

    with pytest.raises(ValueError, match="Proximal should take 2 arguments"):
        cuqi.implicitprior.RegularizedGaussian(np.zeros(5), 1, proximal=lambda s: s)

    # Projector
    with pytest.raises(ValueError, match="Projector needs to be callable"):
        cuqi.implicitprior.RegularizedGaussian(np.zeros(5), 1, projector=1)

    with pytest.raises(ValueError, match="Projector should take 1 argument"):
        cuqi.implicitprior.RegularizedGaussian(np.zeros(5), 1, projector=lambda s,z: s)

def test_ConstrainedGaussian_alias():
    """ Test that the implicit constrained Gaussian is a correct allias for an implicit regularized Gaussian """

    x = cuqi.implicitprior.ConstrainedGaussian(np.zeros(5), 1, constraint="nonnegativity")

    assert isinstance(x, cuqi.implicitprior.RegularizedGaussian)
    assert x.preset == "nonnegativity"

def test_NonnegativeGaussian_alias():
    """ Test that the implicit nonnegative Gaussian is a correct allias for an implicit regularized Gaussian """

    x = cuqi.implicitprior.NonnegativeGaussian(np.zeros(5), 1)

    assert isinstance(x, cuqi.implicitprior.RegularizedGaussian)
    assert x.preset == "nonnegativity"

def test_ConstrainedGMRF_alias():
    """ Test that the implicit constrained GMRF is a correct allias for an implicit regularized GMRF """

    x = cuqi.implicitprior.ConstrainedGMRF(np.zeros(5), 1, constraint="nonnegativity")

    assert isinstance(x, cuqi.implicitprior.RegularizedGMRF)
    assert x.preset == "nonnegativity"

def test_NonnegativeGMRF_alias():
    """ Test that the implicit nonnegative GMRF is a correct allias for an implicit regularized GMRF """

    x = cuqi.implicitprior.NonnegativeGMRF(np.zeros(5), 1)

    assert isinstance(x, cuqi.implicitprior.RegularizedGMRF)
    assert x.preset == "nonnegativity"

def test_RegularizedUnboundedUniform_is_RegularizedGaussian():
    """ Test that the implicit regularized unbounded uniform create a Regularized Gaussian with zero sqrtprec """
    # NOTE: Test is based on the current assumption that the regularized uniform is modeled as a Gaussian with zero precision. This might change in the future.

    x = cuqi.implicitprior.RegularizedUnboundedUniform(cuqi.geometry.Continuous1D(5), regularization="l1", strength = 5.0)
    
    assert np.allclose(x.gaussian.sqrtprec, 0.0)
