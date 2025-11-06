import pytest
import cuqi
import numpy as np
from cuqi.algebra._abstract_syntax_tree import VariableNode
from cuqi.algebra import RandomVariable

def test_randomvariable_name_consistency():

    # Case 1: distribution with explicit name
    x_rv = RandomVariable(cuqi.distribution.Gaussian(0, 1, name="x"))

    assert x_rv.name == "x"

    # Case 2: distribution with implicit name
    x = RandomVariable(cuqi.distribution.Gaussian(0, 1))

    assert x.name == "x"


def test_algrabraic_operations_on_distribution_should_create_randomvariable():

    x = RandomVariable(cuqi.distribution.Gaussian(0, 1))

    assert isinstance(x+1, RandomVariable)
    assert isinstance(1+x, RandomVariable)

    assert isinstance(x-1, RandomVariable)
    assert isinstance(1-x, RandomVariable)

    assert isinstance(x*1, RandomVariable)
    assert isinstance(1*x, RandomVariable)

    assert isinstance(x/1, RandomVariable)
    assert isinstance(1/x, RandomVariable)

    assert isinstance(x**1, RandomVariable)

    assert isinstance(x@1, RandomVariable)
    assert isinstance(1@x, RandomVariable)

    assert isinstance(-x, RandomVariable)

    assert isinstance(abs(x), RandomVariable)

    assert isinstance(x[0], RandomVariable)


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
def test_algebra_on_randomvariables_can_be_computed_and_is_correct(operations):
    x = RandomVariable(cuqi.distribution.Gaussian(np.zeros(2), 1))
    rv = operations(x)
    val = np.random.randn(2)
    # Compare random variable recorded operations vs actual operations
    assert np.allclose(rv(val), operations(val))

def test_randomvariable_returns_correct_parameter_name():
    z = RandomVariable(cuqi.distribution.Gaussian(0, 1))
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
    x = RandomVariable(cuqi.distribution.Gaussian(np.zeros(2), 1))
    y = RandomVariable(cuqi.distribution.Gaussian(np.zeros(2), operations(x)))

    # Condition y on a random fixed value of x
    val = np.random.randn(2)
    y_cond_x = y.condition(x=val)

    # Check basic classes are correct
    assert isinstance(y, RandomVariable)
    assert isinstance(y.distribution.cov, RandomVariable)

    # Check conditioning works and provides the expected result
    assert np.allclose(y_cond_x.distribution.cov, operations(val))

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
    x = RandomVariable(cuqi.distribution.Gaussian(0, 1))
    y = RandomVariable(cuqi.distribution.Gaussian(0, 1))
    z = RandomVariable(cuqi.distribution.Gaussian(0, 1))

    # Define a random variable in joint space
    rv = operations(x, y, z)

    # Check that the random variable is a random variable
    assert isinstance(rv, RandomVariable)

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
    x = RandomVariable(cuqi.distribution.Gaussian(0, 1))
    y = RandomVariable(cuqi.distribution.Gaussian(0, 1))
    z = RandomVariable(cuqi.distribution.Gaussian(0, 1))

    # Define a random variable in joint space
    rv = operations(x, y, z)

    # Fix rng and sample
    np.random.seed(0)
    result = rv.sample()

    # Check that the result compares to the expected result
    np.random.seed(0)
    expected_result = operations(x.sample(), y.sample(), z.sample())

    assert np.allclose(result, expected_result)

@pytest.mark.xfail(reason="logd method for random variable not yet implemented")
def test_logp_conditional():
    """ This tests logp evaluation for conditional random variables """
    # Base example logp value
    true_val = cuqi.distribution.Gaussian(3, 7).logd(13)

    # Distribution with no specified parameters
    x = RandomVariable(cuqi.distribution.Gaussian(cov=lambda s:s, geometry=1))

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

    x = RandomVariable(cuqi.distribution.Gaussian(0, 1, geometry=10))

    assert x.dim == 10
    assert isinstance(x.geometry, cuqi.geometry._DefaultGeometry1D)
    assert x.name == "x"
    assert isinstance(x.tree, VariableNode)
    assert x.tree.name == "x"
    assert isinstance(x.distribution, cuqi.distribution.Gaussian)

def test_rv_variable_name_detection():
    """Test that the rv name is detected correctly at different levels of the python stack. """

    # Test that the density variable name is detected correctly at current level.
    x = RandomVariable(cuqi.distribution.Gaussian(geometry=1))
    assert x.name == 'x'

    # Test that variable name is detected correctly 1 level deep.
    def inner_name():
        y = RandomVariable(cuqi.distribution.Gaussian(geometry=1))
        assert y.name == 'y'
    inner_name()

    # Test variable name is detected correctly at n levels deep.
    class recursive_name:
        def __init__(self, max_recursion=10):
            self.max_recursion = max_recursion
        def __call__(self, current_recursion=0):
            if current_recursion == self.max_recursion:
                z = RandomVariable(cuqi.distribution.Gaussian(geometry=1))
                assert z.name == 'z'
            else:
                self(current_recursion + 1)
    recursive_name()()

def test_variable_name_accross_frames():
    """ Test variable name can be inferred across multiple stack frames. """

    h = RandomVariable(cuqi.distribution.Gaussian(geometry=1)) # Name should be 'h'

    def recursive_return_rv(rv, recursions):
        if recursions == 0:
            assert rv.name == 'h' # h was defined many frames above, and name should be inferred correctly.
        else:
            recursive_return_rv(rv, recursions - 1)
    
    recursive_return_rv(h, 10)

def test_rv_name_consistency():

    x = RandomVariable(cuqi.distribution.Gaussian(geometry=1))
    x2 = x.condition(mean=1)
    x3 = x2.condition(cov=1)

    # Names should be the same as the original density.  
    assert x3.name == 'x'
    assert x2.name == 'x' 
    assert x.name == 'x'

    # Ensure that the name cannot be changed for conditioned densities.
    with pytest.raises(ValueError, match=r"This random variable is derived from the conditional random variable named x"):
        x2.name = 'y'

    x.name = 'y'

    # Ensure that the name is changed for the other conditioned densities.
    assert x2.name == 'y'
    assert x3.name == 'y'

def test_RV_should_catch_non_linear_model_used_as_linear_model():
    A = cuqi.testproblem.Poisson1D().model
    x = RandomVariable(cuqi.distribution.Gaussian(0, 1, geometry=A.domain_geometry))

    with pytest.raises(TypeError, match=r"Cannot apply matmul to non-linear models"):
        y=A@x

def test_ensure_that_RV_evaluation_requires_all_parameters():
    x = RandomVariable(cuqi.distribution.Gaussian(0, 1))

    # raise ValueError(f"Expected {self.parameter_names} arguments, got {kwargs}")
    with pytest.raises(ValueError, match=r"Expected arguments \['x'\], got arguments"):
        x()

    with pytest.raises(ValueError, match=r"Expected arguments \['x'\], got arguments \{'y': 1\}"):
        x(y=1)

    with pytest.raises(ValueError, match=r"Expected arguments \['x'\], got arguments \{'x': 1, 'y': 1\}"):
        x(x=1, y=1)

    y = RandomVariable(cuqi.distribution.Gaussian(0, 1))
    z = (x+y)**2

    with pytest.raises(ValueError, match=r"Expected arguments \['x', 'y'\], got arguments \{'x': 1\}"):
        z(x=1)

def test_RV_sets_name_of_internal_conditional_density_if_par_name_not_set_and_does_not_set_original_density():
    # Case 1
    z = RandomVariable(cuqi.distribution.Gaussian(0, lambda s: s))

    assert z.name == 'z'
    assert z.distribution.name == "z"

    # Case 2 (conditioned density. Should not be able to set name here)
    z = RandomVariable(cuqi.distribution.Gaussian(0, lambda s: s)(s=3))

    assert z.name == 'z'
    assert z.distribution.name == "z"

def test_RV_condition_maintains_parameter_name_order():

    x = RandomVariable(cuqi.distribution.Gaussian(0, lambda s: s))
    y = RandomVariable(cuqi.distribution.Gaussian(0, lambda d: d))

    z = x+y

    assert z.parameter_names == ['x', 'y']
    assert z.condition(s=1).parameter_names == ['x', 'y']
    assert z.condition(d=1).parameter_names == ['x', 'y']
    assert z.condition(d=1, s=1).parameter_names == ['x', 'y']

def test_equivalent_ways_to_create_RV_from_distribution():
    x = RandomVariable(cuqi.distribution.Gaussian(0, 1))
    y = cuqi.distribution.Gaussian(0, 1).rv

    assert x.dim == y.dim
    assert x.distribution.mean == y.distribution.mean
    assert x.distribution.cov == y.distribution.cov

    x = RandomVariable(cuqi.distribution.Gaussian(0, lambda s: s))
    y = cuqi.distribution.Gaussian(0, lambda s: s).rv

    assert x.dim == y.dim
    assert x.distribution.mean == y.distribution.mean
    assert x.condition(s=1).distribution.cov == y.condition(s=1).distribution.cov

def test_RV_sampling_unable_if_conditioning_variables_from_lambda():
    x = RandomVariable(cuqi.distribution.Gaussian(0, lambda s: s))

    with pytest.raises(NotImplementedError, match=r"Unable to directly sample from a random variable that has distributions with conditioning variables"):
        x.sample()

    x.condition(s=1).sample() # This should work

def test_RV_sampling_unable_if_conditioning_variables_from_RV():

    x = RandomVariable(cuqi.distribution.Gaussian(0, 1))
    y = RandomVariable(cuqi.distribution.Gaussian(x, 1))

    with pytest.raises(NotImplementedError, match=r"Unable to directly sample from a random variable that has distributions with conditioning variables"):
        y.sample() # One might expect this to work, but it is not implemented at this time.

def test_RV_sample_against_distribution_sample():
    x = RandomVariable(cuqi.distribution.Gaussian(np.zeros(2), 1))

    np.random.seed(0)
    rv_samples = x.sample(10)

    np.random.seed(0)
    dist_samples = x.distribution.sample(10)

    assert np.allclose(rv_samples.samples, dist_samples.samples)

