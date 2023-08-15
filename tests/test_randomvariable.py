import pytest
import cuqi
import numpy as np

def test_randomvariable_should_require_named_distribution():

    # Case 1: distribution with explicit name
    x_rv = cuqi.randomvariable.RandomVariable(cuqi.distribution.Gaussian(0, 1, name="x"))

    # Case 2: distribution with implicit name
    x = cuqi.distribution.Gaussian(0, 1)
    x_rv = cuqi.randomvariable.RandomVariable(x)

    # Case 3: distribution without name
    with pytest.raises(ValueError, match="without name"):
        x_rv = cuqi.randomvariable.RandomVariable(cuqi.distribution.Gaussian(0, 1))

def test_algrabraic_operations_on_distribution_should_create_randomvariable():

    x = cuqi.distribution.Gaussian(0, 1)

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
    X = cuqi.distribution.Gaussian(np.zeros(2), 1)
    rv = operations(X)
    val = np.random.randn(2)
    # Compare random variable recorded operations vs actual operations
    assert np.allclose(rv(val), operations(val))

def test_randomvariable_returns_correct_parameter_name():
    z = cuqi.distribution.Gaussian(0, 1)
    assert cuqi.utilities.get_non_default_args(z._as_random_variable()) == ["z"]

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
    x = cuqi.distribution.Gaussian(np.zeros(2), 1)
    y = cuqi.distribution.Gaussian(np.zeros(2), operations(x))

    # Condition y on a random fixed value of x
    val = np.random.randn(2)
    y_cond_x = y(x=val)

    # Check basic classes are correct
    assert isinstance(y, cuqi.distribution.Distribution)
    assert isinstance(y.cov, cuqi.randomvariable.RandomVariable)

    # Check conditioning works and provides the expected result
    assert np.allclose(y_cond_x.cov, operations(val))

@pytest.mark.parametrize("operations", [
    lambda x, y, z: x + y,
    lambda x, y, z: x * y,
    lambda x, y, z: x / y,
    lambda x, y, z: x ** y,
    lambda x, y, z: x @ y,
    lambda x, y, z: x + y + z,
    lambda x, y, z: x * y * z,
    lambda x, y, z: x / y / z,
    lambda x, y, z: x ** y ** z,
    lambda x, y, z: x + y * z,
    lambda x, y, z: x * y + z,
    lambda x, y, z: x + y @ z,
    lambda x, y, z: x @ y + z,
    lambda x, y, z: x[0]+z,
    lambda x, y, z: abs(x[0])+z,
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
    x = cuqi.distribution.Gaussian(0, 1)
    y = cuqi.distribution.Gaussian(0, 1)
    z = cuqi.distribution.Gaussian(0, 1)

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
    x = cuqi.distribution.Gaussian(0, 1)
    y = cuqi.distribution.Gaussian(0, 1)
    z = cuqi.distribution.Gaussian(0, 1)

    # Define a random variable in joint space
    rv = operations(x, y, z)

    # Fix rng and sample
    np.random.seed(0)
    result = rv.sample()

    # Check that the result compares to the expected result
    np.random.seed(0)
    expected_result = operations(x.sample(), y.sample(), z.sample())

    assert np.allclose(result, expected_result)

