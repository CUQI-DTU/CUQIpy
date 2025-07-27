import cuqi
import numpy as np
from cuqi.array import CUQIarray
from cuqi.geometry import Continuous1D
from tests.test_model import MultipleInputTestModel

# Get the failing test case
test_model = MultipleInputTestModel()
test_cases = test_model.create_model_test_case_combinations()
test_model9, test_data9 = test_cases[9]

print('Test model 9:')
print(f'  Model type: {type(test_model9)}')
print(f'  Domain geometry: {test_model9.domain_geometry}')
print(f'  Range geometry: {test_model9.range_geometry}')

print('\nTest data 9:')
print(f'  Direction type: {type(test_data9.direction)}')
print(f'  Direction: {test_data9.direction}')
print(f'  Forward input:')
for k, v in test_data9.forward_input.items():
    print(f'    {k}: {type(v)} = {v}')
print(f'  Expected grad output type: {test_data9.expected_grad_output_type}')

# Let's check what the gradient method is doing internally
print('\nDebugging gradient method...')

# Get the kwargs that would be passed to the gradient method
kwargs = test_data9.forward_input.copy()

# Check if any inputs are CUQIarray
is_direction_cuqiarray = isinstance(test_data9.direction, CUQIarray)
is_any_input_cuqiarray = any(isinstance(x, CUQIarray) for x in kwargs.values())

print(f'  Is direction CUQIarray: {is_direction_cuqiarray}')
print(f'  Is any input CUQIarray: {is_any_input_cuqiarray}')

# This is what the gradient method should set
to_CUQIarray = is_direction_cuqiarray or is_any_input_cuqiarray
print(f'  to_CUQIarray should be: {to_CUQIarray}')

print('\nComputing gradient...')
result = test_model9.gradient(test_data9.direction, **test_data9.forward_input)

print(f'\nResult:')
print(f'  Type: {type(result)}')
print(f'  Value: {result}')
print(f'  Is expected type: {isinstance(result, test_data9.expected_grad_output_type)}')

# Check each value in the result dictionary
if isinstance(result, dict):
    print(f'\nResult dictionary values:')
    for k, v in result.items():
        print(f'  {k}: {type(v)} = {v}')
        print(f'    Is CUQIarray: {isinstance(v, cuqi.array.CUQIarray)}') 