import cuqi
import pytest
import numpy as np

def test_density_variable_name_detection():
    """Test that the density variable name is detected correctly at different levels of the python stack. """

    # Test that the density variable name is detected correctly at current level.
    x = cuqi.distribution.Gaussian(geometry=1).rv
    assert x.dist.par_name == 'x'

    # Test that variable name is detected correctly 1 level deep.
    def inner_name():
        y = cuqi.distribution.Gaussian(geometry=1).rv
        assert y.dist.par_name == 'y'
    inner_name()

    # Test variable name is detected correctly at n levels deep.
    class recursive_name:
        def __init__(self, max_recursion=10):
            self.max_recursion = max_recursion
        def __call__(self, current_recursion=0):
            if current_recursion == self.max_recursion:
                z = cuqi.distribution.Gaussian(geometry=1).rv
                assert z.dist.par_name == 'z'
            else:
                self(current_recursion + 1)
    recursive_name()()

def test_variable_name_accross_frames():
    """ Test variable name can be inferred across multiple stack frames. """

    h = cuqi.distribution.Gaussian(geometry=1).rv # Name should be 'h'

    def recursive_return_rv(rv, recursions):
        if recursions == 0:
            assert rv.dist.par_name == 'h' # h was defined many frames above, and name should be inferred correctly.
        else:
            recursive_return_rv(rv, recursions - 1)
    
    recursive_return_rv(h, 10)

def test_density_name_consistency():

    x = cuqi.distribution.Gaussian(geometry=1).rv
    X = x.dist
    X2 = X(mean=1)
    X3 = X2(cov=1)

    # Names should be the same as the original density.  
    assert X3.par_name == 'x'
    assert X2.par_name == 'x' 
    assert X.par_name == 'x'

    # Ensure that the name cannot be changed for conditioned densities.
    with pytest.raises(ValueError, match=r"This density is derived from the conditional density named x."):
        X2.par_name = 'y'

    X.par_name = 'y'

    # Ensure that the name is changed for the other conditioned densities.
    assert X2.par_name == 'y'
    assert X3.par_name == 'y'

def test_evaluated_density_gradient():
    """ Test that the gradient of the evaluated density is not implemented. """
    x = cuqi.distribution.Gaussian(np.zeros(2), np.eye(2))
    x = x(np.zeros(2)+.1)
    with pytest.raises(NotImplementedError, match=r"gradient is not implemented"):
        x.gradient()

def test_allow_setting_par_name_of_unnamed_conditioned_density():
    """ Test that the name of an unnamed conditioned density can be set. """
    Z_s = cuqi.distribution.Gaussian(0, lambda s: s)
    Z = Z_s(s=3)
    Z.par_name = 'Z' # Should not raise an error.

    assert Z_s.par_name == 'Z' # Should be changed for the original density.
