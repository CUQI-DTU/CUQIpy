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
