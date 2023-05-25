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
def test_algebra_on_randomvariables_can_be_combined(operations):
    X = cuqi.distribution.Gaussian(np.zeros(2), 1)
    rv = operations(X)
    val = np.random.randn(2)
    assert np.allclose(rv(val), operations(val))

def test_randomvariable_returns_correct_parameter_name():
    z = cuqi.distribution.Gaussian(0, 1)
    assert cuqi.utilities.get_non_default_args(z.as_random_variable()) == ["z"]




