from tests.test_model import MultipleInputTestModel
import cuqi
import numpy as np

# Get the failing test case
test_model = MultipleInputTestModel()
test_cases = test_model.create_model_test_case_combinations()
test_model50, test_data50 = test_cases[50]

print('Test model 50:')
print(f'  Model type: {type(test_model50)}')
print(f'  Domain geometry: {test_model50.domain_geometry}')
print(f'  Range geometry: {test_model50.range_geometry}')
print(f'  Has gradient func: {test_model50._gradient_func is not None}')
print(f'  Gradient func type: {type(test_model50._gradient_func)}')
print(f'  Has _do_test_gradient: {hasattr(test_model50, "_do_test_gradient")}')
if hasattr(test_model50, "_do_test_gradient"):
    print(f'  _do_test_gradient: {test_model50._do_test_gradient}')

# Let's check all model variations for the steady state PDE model
print('\nChecking all model variations for steady state PDE model:')
test_model_pde = MultipleInputTestModel.helper_build_steady_state_PDE_test_model()
test_model_pde.populate_model_variations()

for i, model in enumerate(test_model_pde.model_variations):
    print(f'  Model variation {i}:')
    print(f'    Type: {type(model)}')
    print(f'    Has gradient func: {model._gradient_func is not None}')
    print(f'    Has _do_test_gradient: {hasattr(model, "_do_test_gradient")}')
    if hasattr(model, "_do_test_gradient"):
        print(f'    _do_test_gradient: {model._do_test_gradient}')
    print()

# Check the filtering logic
print('Checking filtering logic:')
from tests.test_model import model_test_case_combinations_no_gradient_error

# Find which test case corresponds to test_model50
for i, (model, data) in enumerate(model_test_case_combinations_no_gradient_error):
    if model is test_model50:
        print(f'Test model 50 is at index {i} in the filtered list')
        break
else:
    print('Test model 50 is NOT in the filtered list - this is the problem!')

# Check if test_model50 should be filtered out
print(f'\nShould test_model50 be filtered out?')
print(f'  Has _do_test_gradient: {hasattr(test_model50, "_do_test_gradient")}')
if hasattr(test_model50, "_do_test_gradient"):
    print(f'  _do_test_gradient value: {test_model50._do_test_gradient}')
    print(f'  Should be filtered: {not test_model50._do_test_gradient}')
else:
    print(f'  Should be filtered: False (no _do_test_gradient attribute)')

# Check the original test cases vs filtered test cases
print(f'\nChecking test case counts:')
print(f'  Original test cases: {len(test_cases)}')
print(f'  Filtered test cases: {len(model_test_case_combinations_no_gradient_error)}')

# Check which test cases are being used in the failing test
print(f'\nChecking which test cases are in the filtered list:')
for i, (model, data) in enumerate(model_test_case_combinations_no_gradient_error):
    if isinstance(model, cuqi.model._model.PDEModel):
        print(f'  Index {i}: PDEModel with gradient: {model._gradient_func is not None}')
        if hasattr(model, "_do_test_gradient"):
            print(f'    _do_test_gradient: {model._do_test_gradient}')

# The issue might be that the test is somehow using the wrong test case
# Let me check if there's a mismatch between the test case generation and the test execution
print(f'\nChecking if test_model50 is actually one of the model variations:')
for i, model in enumerate(test_model_pde.model_variations):
    if model is test_model50:
        print(f'  test_model50 is model variation {i}')
        break
else:
    print('  test_model50 is NOT one of the model variations - this is strange!')

# Let me check if there's an issue with the test case generation
print(f'\nChecking test case generation:')
print(f'  test_model50 id: {id(test_model50)}')
print(f'  test_model50 domain geometry: {test_model50.domain_geometry}')
print(f'  test_model50 range geometry: {test_model50.range_geometry}')

# Check if test_model50 matches any of the model variations by comparing properties
print(f'\nChecking if test_model50 matches any model variation by properties:')
for i, model in enumerate(test_model_pde.model_variations):
    if (isinstance(model, type(test_model50)) and 
        str(model.domain_geometry) == str(test_model50.domain_geometry) and
        str(model.range_geometry) == str(test_model50.range_geometry)):
        print(f'  test_model50 matches model variation {i} by properties')
        print(f'    Model variation {i} has gradient: {model._gradient_func is not None}')
        print(f'    test_model50 has gradient: {test_model50._gradient_func is not None}')
        break
else:
    print('  test_model50 does not match any model variation by properties')

# Let me check if there are multiple test models being created
print(f'\nChecking if there are multiple test models:')
test_models = []
for i in range(4):  # Check the first 4 test models
    test_model_i = MultipleInputTestModel.helper_build_steady_state_PDE_test_model()
    test_model_i.populate_model_variations()
    test_models.append(test_model_i)

print(f'  Number of test models created: {len(test_models)}')
for i, tm in enumerate(test_models):
    print(f'  Test model {i} has {len(tm.model_variations)} variations')
    for j, model in enumerate(tm.model_variations):
        if (isinstance(model, type(test_model50)) and 
            str(model.domain_geometry) == str(test_model50.domain_geometry) and
            str(model.range_geometry) == str(test_model50.range_geometry) and
            not model._gradient_func):
            print(f'    Found matching model at variation {j} in test model {i}')

print('\nTest data 50:')
print(f'  Direction type: {type(test_data50.direction)}')
print(f'  Direction: {test_data50.direction}')
print(f'  Forward input:')
for k, v in test_data50.forward_input.items():
    print(f'    {k}: {type(v)} = {v}')

# Test the gradient computation
print('\nTesting gradient computation...')

# Par input
par_input = test_data50.forward_input
try:
    grad_output_par_input = test_model50.gradient(test_data50.direction, **par_input)
    print("Par input gradient computation successful")
except Exception as e:
    print(f"Par input gradient computation failed: {e}")
    grad_output_par_input = None

# Fun input
fun_input = {
    k: v.funvals if (isinstance(v, cuqi.array.CUQIarray) or isinstance(v, cuqi.samples.Samples)) else v
    for k, v in par_input.items()
}

# Check if any inputs are function values (extracted from CUQIarray)
has_funvals = any(
    isinstance(par_input[k], (cuqi.array.CUQIarray, cuqi.samples.Samples))
    for k in fun_input.keys()
)
is_var_par = not has_funvals  # If we have funvals, set is_var_par=False

print(f'\nHas funvals: {has_funvals}')
print(f'is_var_par: {is_var_par}')

print('\nPar input:')
for k, v in par_input.items():
    print(f'  {k}: {type(v)} = {v}')

print('\nFun input:')
for k, v in fun_input.items():
    print(f'  {k}: {type(v)} = {v}')

try:
    grad_output_fun_input = test_model50.gradient(test_data50.direction, is_var_par=is_var_par, **fun_input)
    print("Fun input gradient computation successful")
except Exception as e:
    print(f"Fun input gradient computation failed: {e}")
    grad_output_fun_input = None

if grad_output_par_input is not None and grad_output_fun_input is not None:
    print('\nGradient outputs:')
    print('Par input gradient:')
    for k, v in grad_output_par_input.items():
        print(f'  {k}: {type(v)} = {v}')

    print('\nFun input gradient:')
    for k, v in grad_output_fun_input.items():
        print(f'  {k}: {type(v)} = {v}')

    # Check if they're close
    print('\nComparison:')
    for k, v in grad_output_par_input.items():
        if v is not None:
            if isinstance(v, cuqi.array.CUQIarray):
                v_par = v.to_numpy()
            else:
                v_par = v
                
            if isinstance(grad_output_fun_input[k], cuqi.array.CUQIarray):
                v_fun = grad_output_fun_input[k].to_numpy()
            else:
                v_fun = grad_output_fun_input[k]
                
            print(f'  {k}:')
            print(f'    Par: {v_par}')
            print(f'    Fun: {v_fun}')
            print(f'    Close: {np.allclose(v_par, v_fun)}')
            print(f'    Diff: {np.abs(v_par - v_fun)}') 