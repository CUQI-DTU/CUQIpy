import cuqi
import pytest
import numpy as np

def test_density_variable_name_detection():
    """Test that the density variable name is detected correctly at different levels of the python stack. """

    # Test that the density variable name is detected correctly at current level.
    x = cuqi.distribution.Gaussian(geometry=1)
    assert x.name == 'x'

    # Test that variable name is detected correctly 1 level deep.
    def inner_name():
        y = cuqi.distribution.Gaussian(geometry=1)
        assert y.name == 'y'
    inner_name()

    # Test variable name is detected correctly at n levels deep.
    class recursive_name:
        def __init__(self, max_recursion=10):
            self.max_recursion = max_recursion
        def __call__(self, current_recursion=0):
            if current_recursion == self.max_recursion:
                z = cuqi.distribution.Gaussian(geometry=1)
                assert z.name == 'z'
            else:
                self(current_recursion + 1)
    recursive_name()()

def test_variable_name_accross_frames():
    """ Test variable name can be inferred across multiple stack frames. """

    h = cuqi.distribution.Gaussian(geometry=1) # Name should be 'h'

    def recursive_return_dist(dist, recursions):
        if recursions == 0:
            assert dist.name == 'h' # h was defined many frames above, and name should be inferred correctly.
        else:
            recursive_return_dist(dist, recursions - 1)
    
    recursive_return_dist(h, 10)

def test_density_name_consistency():

    x = cuqi.distribution.Gaussian(geometry=1)
    x2 = x(mean=1)
    x3 = x2(cov=1)

    # Names should be the same as the original density.  
    assert x3.name == 'x'
    assert x2.name == 'x' 
    assert x.name == 'x'

    # Ensure that the name cannot be changed for conditioned densities.
    with pytest.raises(ValueError, match=r"Cannot set name of conditioned density. Only the original density can have its name set."):
        x2.name = 'y'

    x.name = 'y'

    # Ensure that the name is changed for the other conditioned densities.
    assert x2.name == 'y'
    assert x3.name == 'y'

def test_evaluated_density_gradient():
    """ Test that the gradient of the evaluated density is not implemented. """
    x = cuqi.distribution.Gaussian(np.zeros(2), np.eye(2))
    x = x(np.zeros(2)+.1)
    with pytest.raises(NotImplementedError, match=r"gradient is not implemented"):
        x.gradient()
