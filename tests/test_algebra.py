import pytest
import numpy as np
from cuqi.algebra import VariableNode


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
def test_algebra_on_single_variables_is_correct(operations):
    """ Tests that algebraic operations on variables are recorded correctly by comparing to the lambda expression """
    X = VariableNode('x')
    rv = operations(X)
    val = np.random.randn(2)
    # Compare random variable recorded operations vs actual operations
    assert np.allclose(rv(x=val), operations(val))

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
def test_algebra_on_joint_variables_is_correct(operations):
    """ Tests that algebraic operations on variables are recorded correctly by comparing to the lambda expression """
    x = VariableNode('x')
    y = VariableNode('y')
    z = VariableNode('z')

    # Define a variable in joint space
    rv = operations(x, y, z)

    # Check operations work
    val_x = abs(np.random.randn(2))+1
    val_y = abs(np.random.randn(2))+1
    val_z = abs(np.random.randn(2))+1
    assert np.allclose(rv(x=val_x, y=val_y, z=val_z), operations(val_x, val_y, val_z))
