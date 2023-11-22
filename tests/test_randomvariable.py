import pytest
import cuqi
import numpy as np
from cuqi.randomvariable._ast import RandomVariableNode

def test_randomvariable_name_consistency():

    # Case 1: distribution with explicit name
    x_rv = cuqi.randomvariable.RandomVariable(cuqi.distribution.Gaussian(0, 1, par_name="x"))

    assert x_rv.name == "x"

    # Case 2: distribution with implicit name
    x = cuqi.distribution.Gaussian(0, 1).rv

    assert x.name == "x"


def test_algrabraic_operations_on_distribution_should_create_randomvariable():

    x = cuqi.distribution.Gaussian(0, 1).rv

    assert isinstance(x+1, cuqi.randomvariable.RandomVariable)
    assert isinstance(1+x, cuqi.randomvariable.RandomVariable)

    assert isinstance(x-1, cuqi.randomvariable.RandomVariable)
    assert isinstance(1-x, cuqi.randomvariable.RandomVariable)

    assert isinstance(x*1, cuqi.randomvariable.RandomVariable)
    assert isinstance(1*x, cuqi.randomvariable.RandomVariable)

    assert isinstance(x/1, cuqi.randomvariable.RandomVariable)
    assert isinstance(1/x, cuqi.randomvariable.RandomVariable)

    assert isinstance(x**1, cuqi.randomvariable.RandomVariable)

    assert isinstance(x@1, cuqi.randomvariable.RandomVariable)
    assert isinstance(1@x, cuqi.randomvariable.RandomVariable)

    assert isinstance(-x, cuqi.randomvariable.RandomVariable)

    assert isinstance(abs(x), cuqi.randomvariable.RandomVariable)

    assert isinstance(x[0], cuqi.randomvariable.RandomVariable)


@pytest.mark.parametrize("operations", [
    lambda x: x+1,
    lambda x: x**2,
    lambda x: -x,
    lambda x: abs(x),
    lambda x: x/1,
    lambda x: 1/x,
    lambda x: x*1,
    lambda x: 1*x,
    lambda x: 3/abs((x+1)**2-10),
    lambda x: 9/(x+1)**2,
    lambda x: x/3+4,
    lambda x: x[0],
    lambda x: x[0:1],
    lambda x: cuqi.model.LinearModel(np.ones((2,2)))@x,
])
def test_algebra_on_randomvariables_can_be_combined_and_is_correct(operations):
    X = cuqi.distribution.Gaussian(np.zeros(2), 1).rv
    rv = operations(X)
    val = np.random.randn(2)
    # Compare random variable recorded operations vs actual operations
    assert np.allclose(rv(val), operations(val))

def test_randomvariable_returns_correct_parameter_name():
    z = cuqi.distribution.Gaussian(0, 1).rv
    assert cuqi.utilities.get_non_default_args(z) == ["z"]

@pytest.mark.parametrize("operations", [
    lambda x: x+1,
    lambda x: x**2,
    lambda x: -x,
    lambda x: abs(x),
    lambda x: x/1,
    lambda x: 1/x,
    lambda x: x*1,
    lambda x: 1*x,
    lambda x: 3/abs((x+1)**2-10),
    lambda x: 9/(x+1)**2,
    lambda x: x/3+4,
    lambda x: x[0],
    lambda x: x[0:1],
])
def test_randomvariable_works_with_distribution_conditioning(operations):

    # Define x and y | x (y conditioned on x) with some algebraic operations
    x = cuqi.distribution.Gaussian(np.zeros(2), 1).rv
    y = cuqi.distribution.Gaussian(np.zeros(2), operations(x)).rv

    # Condition y on a random fixed value of x
    val = np.random.randn(2)
    y_cond_x = y.condition(x=val)

    # Check basic classes are correct
    assert isinstance(y, cuqi.randomvariable.RandomVariable)
    assert isinstance(y.dist.cov, cuqi.randomvariable.RandomVariable)

    # Check conditioning works and provides the expected result
    assert np.allclose(y_cond_x.dist.cov, operations(val))

@pytest.mark.parametrize("operations", [
    lambda x, y, z: x + y - z,
    lambda x, y, z: x * y - z,
    lambda x, y, z: x / y - z,
    lambda x, y, z: x ** y - z,
    lambda x, y, z: x @ y - z,
    lambda x, y, z: x + y + z,
    lambda x, y, z: x * y * z,
    lambda x, y, z: x / y / z,
    lambda x, y, z: x ** y ** z,
    lambda x, y, z: x + y * z,
    lambda x, y, z: x * y + z,
    lambda x, y, z: x + y @ z,
    lambda x, y, z: x @ y + z,
    lambda x, y, z: x[0]+z+y,
    lambda x, y, z: abs(x[0])+z-y,
    lambda x, y, z: (x + y) * z,
    lambda x, y, z: (x - y) / z,
    lambda x, y, z: x ** (y + z),
    lambda x, y, z: abs(x) + abs(y * z),
    lambda x, y, z: (x + y) * (z + 1) - x * y,
    lambda x, y, z: x / (y ** z) + (x * y) ** z,
    lambda x, y, z: abs(x) * (abs(y) + abs(z)),
    lambda x, y, z: abs(x * y) + abs(y * z) + abs(z * x),
    lambda x, y, z: x * y + y * z + z * x,
])
def test_randomvariable_algebra_works_on_joint_space(operations):
    """ Test that algebraic operations on random variables work in joint space """
    x = cuqi.distribution.Gaussian(0, 1).rv
    y = cuqi.distribution.Gaussian(0, 1).rv
    z = cuqi.distribution.Gaussian(0, 1).rv

    # Define a random variable in joint space
    rv = operations(x, y, z)

    # Check that the random variable is a random variable
    assert isinstance(rv, cuqi.randomvariable.RandomVariable)

    # Check operations work
    val_x = abs(np.random.randn(x.dim))+1
    val_y = abs(np.random.randn(y.dim))+1
    val_z = abs(np.random.randn(z.dim))+1
    assert np.allclose(rv(x=val_x, y=val_y, z=val_z), operations(val_x, val_y, val_z))

@pytest.mark.parametrize("operations", [
    lambda x, y, z: x + y,
    lambda x, y, z: x * y,
    lambda x, y, z: x / y,
    lambda x, y, z: x ** y,
    lambda x, y, z: x + y + z,
    lambda x, y, z: x * y * z,
    lambda x, y, z: x / y / z,
    lambda x, y, z: x ** y ** z,
    lambda x, y, z: x + y * z,
    lambda x, y, z: x * y + z,
    lambda x, y, z: (x + y) * z,
    lambda x, y, z: (x - y) / z,
    lambda x, y, z: x ** (y + z),
    lambda x, y, z: abs(x) + abs(y * z),
    lambda x, y, z: (x + y) * (z + 1) - x * y,
    lambda x, y, z: x / (y ** z) + (x * y) ** z,
    lambda x, y, z: abs(x) * (abs(y) + abs(z)),
    lambda x, y, z: abs(x * y) + abs(y * z) + abs(z * x),
    lambda x, y, z: x * y + y * z + z * x,
])
def test_randomvariable_sample(operations):
    """ Test that random variable sampling works """
    x = cuqi.distribution.Gaussian(0, 1).rv
    y = cuqi.distribution.Gaussian(0, 1).rv
    z = cuqi.distribution.Gaussian(0, 1).rv

    # Define a random variable in joint space
    rv = operations(x, y, z)

    # Fix rng and sample
    np.random.seed(0)
    result = rv.sample()

    # Check that the result compares to the expected result
    np.random.seed(0)
    expected_result = operations(x.sample(), y.sample(), z.sample())

    assert np.allclose(result, expected_result)

def test_logp_conditional():
    """ This tests logp evaluation for conditional random variables """
    # Base example logp value
    true_val = cuqi.distribution.Gaussian(3, 7).logd(13)

    # Distribution with no specified parameters
    x = cuqi.distribution.Gaussian(cov=lambda s:s, geometry=1).rv

    # Test logp evaluates correctly in various cases
    assert x.logd(mean=3, s=7, x=13) == true_val
    assert x.condition(x=13).logd(mean=3, s=7) == true_val
    assert x.condition(x=13, mean=3).logd(s=7) == true_val
    assert x.condition(x=13, mean=3, s=7).logd() == true_val
    assert x.condition(mean=3).logd(s=7, x=13) == true_val
    assert x.condition(mean=3, s=7).logd(x=13) == true_val
    assert x.condition(mean=3, x=13).logd(s=7) == true_val

def test_rv_attributes():
    """ Test various attributes of random variable"""

    x = cuqi.distribution.Gaussian(0, 1, geometry=10).rv

    assert x.dim == 10
    assert isinstance(x.geometry, cuqi.geometry._DefaultGeometry1D)
    assert x.name == "x"
    assert isinstance(x.tree, RandomVariableNode)
    assert x.tree.name == "x"
    assert isinstance(x.dist, cuqi.distribution.Gaussian)

def test_rv_variable_name_detection():
    """Test that the rv name is detected correctly at different levels of the python stack. """

    # Test that the density variable name is detected correctly at current level.
    x = cuqi.distribution.Gaussian(geometry=1).rv
    assert x.name == 'x'

    # Test that variable name is detected correctly 1 level deep.
    def inner_name():
        y = cuqi.distribution.Gaussian(geometry=1).rv
        assert y.name == 'y'
    inner_name()

    # Test variable name is detected correctly at n levels deep.
    class recursive_name:
        def __init__(self, max_recursion=10):
            self.max_recursion = max_recursion
        def __call__(self, current_recursion=0):
            if current_recursion == self.max_recursion:
                z = cuqi.distribution.Gaussian(geometry=1).rv
                assert z.name == 'z'
            else:
                self(current_recursion + 1)
    recursive_name()()

def test_variable_name_accross_frames():
    """ Test variable name can be inferred across multiple stack frames. """

    h = cuqi.distribution.Gaussian(geometry=1).rv # Name should be 'h'

    def recursive_return_rv(rv, recursions):
        if recursions == 0:
            assert rv.name == 'h' # h was defined many frames above, and name should be inferred correctly.
        else:
            recursive_return_rv(rv, recursions - 1)
    
    recursive_return_rv(h, 10)

def test_rv_name_consistency():

    x = cuqi.distribution.Gaussian(geometry=1).rv
    x2 = x.condition(mean=1)
    x3 = x2.condition(cov=1)

    # Names should be the same as the original density.  
    assert x3.name == 'x'
    assert x2.name == 'x' 
    assert x.name == 'x'

    # Ensure that the name cannot be changed for conditioned densities.
    with pytest.raises(ValueError, match=r"Cannot set name of conditioned random variable. Only the original variable can have its name set."):
        x2.name = 'y'

    x.name = 'y'

    # Ensure that the name is changed for the other conditioned densities.
    assert x2.name == 'y'
    assert x3.name == 'y'

def test_RV_should_catch_non_linear_model_used_as_linear_model():
    A = cuqi.testproblem.Poisson1D().model
    x = cuqi.distribution.Gaussian(0, 1, geometry=A.domain_geometry).rv

    with pytest.raises(TypeError, match=r"Cannot apply matmul to non-linear models"):
        y=A@x

def test_ensure_that_RV_evaluation_requires_all_parameters():
    x = cuqi.distribution.Gaussian(0, 1).rv

    # raise ValueError(f"Expected {self.parameter_names} arguments, got {kwargs}")
    with pytest.raises(ValueError, match=r"Expected arguments \['x'\], got arguments"):
        x()

    with pytest.raises(ValueError, match=r"Expected arguments \['x'\], got arguments \{'y': 1\}"):
        x(y=1)

    with pytest.raises(ValueError, match=r"Expected arguments \['x'\], got arguments \{'x': 1, 'y': 1\}"):
        x(x=1, y=1)

    y = cuqi.distribution.Gaussian(0, 1).rv
    z = (x+y)**2

    with pytest.raises(ValueError, match=r"Expected arguments \['x', 'y'\], got arguments \{'x': 1\}"):
        z(x=1)

def test_RV_sets_name_of_internal_conditional_density_if_par_name_not_set_and_does_not_set_original_density():
    Z_s = cuqi.distribution.Gaussian(0, lambda s: s)
    Z = Z_s(s=3)
    z = Z.rv

    assert z.name == 'z'
    assert z.dist.par_name == "z"
    assert Z_s.par_name is None # Should not be set for the original density.
    assert Z.par_name is None # Should not be set for the conditioned density.
