from cuqi.array import CUQIarray
from cuqi.geometry import Continuous1D, Discrete
import numpy as np
import scipy as sp
import cuqi
import pytest
from scipy import optimize
from copy import copy, deepcopy
from cuqi.geometry import _identity_geometries
from cuqi.utilities import force_ndarray

class TestMultipleInputModel:
    def __init__(self):
        self.model_class = None
        self.forward_map = None
        self.pde = None
        self.gradient_form1 = None
        self.gradient_form2 = None
        self.gradient_form2_incomplete = None
        self.jacobian_form1 = None
        self.jacobian_form2 = None
        self.jacobian_form2_incomplete = None
        self.domain_geometry = None
        self.range_geometry = None
        self.test_data = []
        self.model_variations = []

    def populate_model_variations(self):
        if self.pde is not None:
            self.populate_pde_model_variations()
        elif self.forward_map is not None:
            self.populate_general_model_variations()

    def populate_general_model_variations(self):
        # model with forward
        model = self.model_class(forward=self.forward_map,
                                  domain_geometry=self.domain_geometry,
                                  range_geometry=self.range_geometry)
        model._test_flag_gradient = False
        self.model_variations.append(model)

        # model with gradient from1
        if self.gradient_form1 is not None:
            model = self.model_class(forward=self.forward_map,
                                      gradient=self.gradient_form1,
                                      domain_geometry=self.domain_geometry,
                                      range_geometry=self.range_geometry)
            self.model_variations.append(model)

        # model with gradient from2
        if self.gradient_form2 is not None:
            model = self.model_class(forward=self.forward_map,
                                      gradient=self.gradient_form2,
                                      domain_geometry=self.domain_geometry,
                                      range_geometry=self.range_geometry)
            self.model_variations.append(model)

        # model with gradient from2 incomplete
        if self.gradient_form2_incomplete is not None:
            model = self.model_class(forward=self.forward_map,
                                        gradient=self.gradient_form2_incomplete,
                                        domain_geometry=self.domain_geometry,
                                        range_geometry=self.range_geometry)
            self.model_variations.append(model)

        # model with jacobian from1
        if self.jacobian_form1 is not None:
            model = self.model_class(forward=self.forward_map,
                                        jacobian=self.jacobian_form1,
                                        domain_geometry=self.domain_geometry,
                                        range_geometry=self.range_geometry)
            self.model_variations.append(model)

        # model with jacobian from2
        if self.jacobian_form2 is not None:
            model = self.model_class(forward=self.forward_map,
                                        jacobian=self.jacobian_form2,
                                        domain_geometry=self.domain_geometry,
                                        range_geometry=self.range_geometry)
            self.model_variations.append(model)

        # model with jacobian from2 incomplete
        if self.jacobian_form2_incomplete is not None:
            model = self.model_class(forward=self.forward_map,
                                        jacobian=self.jacobian_form2_incomplete,
                                        domain_geometry=self.domain_geometry,
                                        range_geometry=self.range_geometry)
            self.model_variations.append(model)

    def populate_pde_model_variations(self):
        # model with PDE
        model = self.model_class(self.pde,
                                 domain_geometry=self.domain_geometry,
                                 range_geometry=self.range_geometry)
        model._test_flag_gradient = False
        self.model_variations.append(model)

    @staticmethod
    def create_model_test_case_combinations():
        model_test_case_combinations = []
        test_model_list = []

        # Model 1
        test_model = TestMultipleInputModel.helper_build_three_input_test_model()
        test_model.populate_model_variations()
        TestCase.create_test_cases_for_test_model(test_model)
        test_model_list.append(test_model)
    
        # Model 2
        test_model = TestMultipleInputModel.helper_build_steady_state_PDE_test_model()
        test_model.populate_model_variations()
        TestCase.create_test_cases_for_test_model(test_model)
        test_model_list.append(test_model)

        # Model 3
        test_model = TestMultipleInputModel.helper_build_two_input_test_with_mapped_parameter_model()
        test_model.populate_model_variations()
        TestCase.create_test_cases_for_test_model(test_model)
        test_model_list.append(test_model) 

        # append all combinations of test model variations and test cases
        # to model_test_case_combinations
        for test_model in test_model_list:
            for test_model_variation in test_model.model_variations:
                for test_data_item in test_model.test_data:
                    model_test_case_combinations.append(
                        (test_model_variation, test_data_item))
        
        return model_test_case_combinations
    
    @staticmethod
    def helper_build_three_input_test_model():
        test_model = TestMultipleInputModel()
        test_model.forward_map = lambda x, y, z: x * y[0] + z * y[1]

        def gradient_x(direction, x, y, z):
            return direction * y[0]
        
        def gradient_y(direction, x, y, z):
            return np.array([direction @ x, direction @ z])
        
        def gradient_z(direction, x, y, z):
            return direction * y[1]
        
        def gradient_form1(direction, x, y, z):
            grad_x = gradient_x(direction, x, y, z)
            grad_y = gradient_y(direction, x, y, z)
            grad_z = gradient_z(direction, x, y, z)
            return (grad_x, grad_y, grad_z)
        
        test_model.gradient_form1 = gradient_form1
        test_model.gradient_form2 = (gradient_x, gradient_y, gradient_z)
        test_model.gradient_form2_incomplete = (gradient_x, None, gradient_z)

        def jacobian_x(x, y, z):
            ones = np.ones_like(x)
            return np.diag(y[0] * ones)
        
        def jacobian_y(x, y, z):
            return np.array([x, z]).T
        
        def jacobian_z(x, y, z):
            ones = np.ones_like(x)
            return np.diag(y[1] * ones)
        
        def jacobian_form1(x, y, z):
            jac_x = jacobian_x(x, y, z)
            jac_y = jacobian_y(x, y, z)
            jac_z = jacobian_z(x, y, z)
            return np.hstack([jac_x, jac_y, jac_z])

        test_model.jacobian_form1 = jacobian_form1
        test_model.jacobian_form2 = (jacobian_x, jacobian_y, jacobian_z)
        test_model.jacobian_form2_incomplete = (None, jacobian_y, jacobian_z)

        test_model.domain_geometry = (cuqi.geometry.Continuous1D(3),
                cuqi.geometry.Continuous1D(2),
                cuqi.geometry.Continuous1D(3))
        test_model.range_geometry = cuqi.geometry.Continuous1D(3)
        test_model.model_class = cuqi.model.Model
        return test_model

    @staticmethod
    def helper_build_steady_state_PDE_test_model():
    # method that builds a multiple input PDE model

        # Poisson equation
        dim = 20 #Number of nodes
        L = 20 # Length of the domain
        dx = L/(dim-1) # grid spacing 
        grid_sol = np.linspace(dx, L, dim-1, endpoint=False)
        grid_obs = grid_sol[5:]
        source =  lambda mag: mag*np.sin(grid_sol) #source term
        kappa = np.ones(dim) #kappa is the diffusivity 
    
        # Build the solver
        FOFD_operator = cuqi.operator.FirstOrderFiniteDifference(dim-1, bc_type='zero', dx=dx).get_matrix().todense()
        diff_operator = lambda kappa_scale: FOFD_operator.T @ np.diag(kappa_scale*kappa) @ FOFD_operator
        poisson_form = lambda mag, kappa_scale: (diff_operator(kappa_scale), source(mag))
        CUQI_pde = cuqi.pde.SteadyStateLinearPDE(poisson_form, grid_sol=grid_sol, grid_obs=grid_obs, observation_map=lambda u:u**2)

        test_model = TestMultipleInputModel()
        test_model.model_class = cuqi.model.PDEModel
        test_model.pde = CUQI_pde
        test_model.domain_geometry = (Discrete(["mag"]), Discrete(['kappa_scale']))
        test_model.range_geometry = Continuous1D(len(grid_obs))
        return test_model

    @staticmethod
    def helper_build_two_input_test_with_mapped_parameter_model():
        """Build a model with two inputs in which the inputs are mapped
        via geometry mapping"""
        test_model = TestMultipleInputModel()
        test_model.forward_map = lambda a, b: np.array([a[0]**2*b[0]*b[1], b[1]*a[0]*a[1]])

        def gradient_a(direction, a, b):
            return np.array([2*a[0]*b[0]*b[1]*direction[0] + b[1]*a[1]*direction[1],
                             b[1]*a[0]*direction[1]])

        def gradient_b(direction, a, b):
            return np.array([a[0]**2*b[1]*direction[0],
                             a[0]**2*b[0]*direction[0]+a[0]*a[1]*direction[1]])

        def gradient_form1(direction, a, b):
            grad_a = gradient_a(direction, a, b)
            grad_b = gradient_b(direction, a, b)
            return (grad_a, grad_b)

        test_model.gradient_form1 = gradient_form1
        test_model.gradient_form2 = (gradient_a, gradient_b)
        test_model.gradient_form2_incomplete = (gradient_a, None)
        test_model.gradient_wrong_signature = None #TODO: Add a test for this
        test_model.gradient_wrong_signature_order = None #TODO: Add a test for this

        test_model.jacobian_form1 = None
        test_model.jacobian_form2 = None
        test_model.jacobian_form2_incomplete = None

        geom_a = cuqi.geometry.MappedGeometry(cuqi.geometry.Continuous1D(2),
                                              map=lambda x: np.sin(x))
        geom_a.gradient = lambda direction, x: direction@np.diag(np.cos(x))

        geom_b = cuqi.geometry.MappedGeometry(cuqi.geometry.Continuous1D(2),
                                              map=lambda x: x**3)
        geom_b.gradient = lambda direction, x: 3*direction@np.diag(x**2)

        test_model.domain_geometry = (geom_a, geom_b)
        test_model.range_geometry = cuqi.geometry.Continuous1D(2)
        test_model.model_class = cuqi.model.Model
        return test_model

class TestCase:
    def __init__(self, test_model):
        self._test_model = test_model
        self._non_default_args = self._test_model.model_variations[0]._non_default_args
        self.forward_input = None
        self.forward_input_stacked = None
        self.direction = None
        self.expected_fwd_output = None
        self.expected_grad_output_value = None
        self.expected_fwd_output_type = None
        self.expected_grad_output_type = None
        self.FD_grad_output = None

    @property
    def model_forward_map(self):
        if self._test_model.pde is not None:
            def _forward(*args, **kwargs):
                # use the model forward map but make sure it maps function
                # values to function value
                value = self._test_model.model_variations[0].forward(
                    *args, is_par=False, **kwargs)
                return self._test_model.range_geometry.par2fun(value)
        else:
            _forward = self._test_model.forward_map
        return _forward

    def create_input(self):
        # create a kwarg dictionary for the inputs
        input_dict = {}
        for i, arg in enumerate(self._non_default_args):
            input_dict[arg] = np.random.randn(self._test_model.domain_geometry[i].par_dim)
        self.forward_input = input_dict
        self.forward_input_stacked = np.hstack([v for v in list(input_dict.values())])
    
    def create_direction(self):
        self.direction = np.random.randn(self._test_model.range_geometry.par_dim)

    def compute_expected_fwd_output(self):
        mapped_input = self._compute_mapped_input(self.forward_input,
                                                  self._test_model.domain_geometry)
        self.expected_fwd_output = self.model_forward_map(**mapped_input)

    @staticmethod
    def _compute_mapped_input(forward_input, domain_geom):
        mapped_input = {}
        for i, (k, v) in enumerate(forward_input.items()):
            if hasattr(domain_geom[i], 'map'):
                mapped_input[k] = domain_geom[i].map(v)
            elif type(domain_geom[i]) in _identity_geometries:
                mapped_input[k] = v
            else:
                raise NotImplementedError("Mapping not implemented for geometry type.")
        return mapped_input

    def compute_expected_grad_output(self):
        # check if all domain geometry components are identity geometries
        domain_geom = self._test_model.domain_geometry
        forward_input_fun = deepcopy(self.forward_input)

        if all([type(geom) in _identity_geometries for geom in domain_geom]):
            self.expected_grad_output_value = self._test_model.gradient_form1(self.direction, **forward_input_fun)
        else:
            for i, (k, v) in enumerate(forward_input_fun.items()):
                forward_input_fun[k] = domain_geom[i].par2fun(v)
            self.expected_grad_output_value = []
            for i, (k, v) in enumerate(self.forward_input.items()):
                self.expected_grad_output_value.append(self._test_model.gradient_form2[i](self.direction, **forward_input_fun))
                self.expected_grad_output_value[-1] = domain_geom[i].gradient(self.expected_grad_output_value[-1], v)
        self.expected_grad_output_value = tuple(self.expected_grad_output_value)

    def compute_FD_grad_output(self):
        FD_grad_list = []
        model = self._test_model.model_variations[0]
        for k, v in self.forward_input.items():
            #forward_input without k, v
            forward_input = self.forward_input.copy()
            del forward_input[k]
            fwd = lambda x: model(**forward_input, **{k:x})
            if isinstance(v, cuqi.array.CUQIarray):
                v = v.to_numpy()
            direction = self.direction
            if isinstance(self.direction, cuqi.array.CUQIarray):
                direction = self.direction.to_numpy()
            FD_grad = cuqi.utilities.approx_derivative(fwd, v, direction)
            FD_grad_list.append(FD_grad)
        self.FD_grad_output = tuple(FD_grad_list)

    @staticmethod
    def create_test_cases_for_test_model(test_model):
        # Case 1: all inputs are numpy arrays
        test_case = TestCase(test_model)
        test_case.create_input()
        test_case.compute_expected_fwd_output()
        test_case.expected_fwd_output_type = np.ndarray

        if test_model.gradient_form1 is not None:
            test_case.create_direction()
            test_case.compute_expected_grad_output()
            test_case.compute_FD_grad_output()
            test_case.expected_grad_output_type = np.ndarray

        test_model.test_data.append(test_case)

        # Case 2: all inputs are CUQIarrays
        test_case = TestCase(test_model)
        test_case.create_input()
        for i, (k, v) in enumerate(test_case.forward_input.items()):
            test_case.forward_input[k] = cuqi.array.CUQIarray(v, geometry=test_model.domain_geometry[i])
        test_case.compute_expected_fwd_output()
        test_case.expected_fwd_output = cuqi.array.CUQIarray(test_case.expected_fwd_output, geometry=test_model.range_geometry)
        test_case.expected_fwd_output_type = cuqi.array.CUQIarray

        if test_model.gradient_form1 is not None:
            test_case.create_direction()
            test_case.direction = cuqi.array.CUQIarray(test_case.direction, geometry=test_model.range_geometry)
            test_case.compute_expected_grad_output()
            test_case.compute_FD_grad_output()
            test_case.expected_grad_output_type = cuqi.array.CUQIarray

        test_model.test_data.append(test_case)

        # Case 3: inputs are mix of CUQIarrays and numpy arrays
        test_case = TestCase(test_model)
        test_case.create_input()
        for i, (k, v) in enumerate(test_case.forward_input.items()):
            if i == 0:
                test_case.forward_input[k] = cuqi.array.CUQIarray(v, geometry=test_model.domain_geometry[i])
        test_case.compute_expected_fwd_output()
        test_case.expected_fwd_output = cuqi.array.CUQIarray(test_case.expected_fwd_output, geometry=test_model.range_geometry)
        test_case.expected_fwd_output_type = cuqi.array.CUQIarray

        if test_model.gradient_form1 is not None:
            test_case.create_direction()
            test_case.compute_expected_grad_output()
            test_case.compute_FD_grad_output()
            test_case.expected_grad_output_type = cuqi.array.CUQIarray

        test_model.test_data.append(test_case)

        # Case 4: same as previous case but direction is a CUQIarray
        test_case = deepcopy(test_case)
        if test_model.gradient_form1 is not None:
            test_case.direction = cuqi.array.CUQIarray(test_case.direction, geometry=test_model.range_geometry)

        test_model.test_data.append(test_case)

        # Case 5: inputs are mix of CUQIarrays and samples # should raise an error
        test_case = TestCase(test_model)
        test_case.create_input()
        for i, (k, v) in enumerate(test_case.forward_input.items()):
            if i == 0:
                v2 = 1.3*v
                samples = np.vstack([v, v2]).T
                test_case.forward_input[k] = cuqi.samples.Samples(samples, geometry=test_model.domain_geometry[i])
            else:
                test_case.forward_input[k] = cuqi.array.CUQIarray(v, geometry=test_model.domain_geometry[i])
        test_case.expected_fwd_output = TypeError(
                "If applying the function to Samples, all inputs should be Samples.")

        if test_model.gradient_form1 is not None:
            test_case.create_direction()
            test_case.expected_grad_output_value = NotImplementedError("Gradient is not implemented for input of type Samples.")

        test_model.test_data.append(test_case)

        # Case 6: inputs are samples # should work for forward but not for gradient
        test_case = TestCase(test_model)
        test_case.create_input()
        for i, (k, v) in enumerate(test_case.forward_input.items()):
            v2 = 1.5*v
            samples = np.vstack([v, v2]).T
            test_case.forward_input[k] = cuqi.samples.Samples(samples, geometry=test_model.domain_geometry[i])
        expected_fwd_output = []
        for i in range(2):
            input_i = {k: v.samples[:,i] for k, v in test_case.forward_input.items()}
            mapped_input_i = TestCase._compute_mapped_input(input_i, 
                                                          test_model.domain_geometry)
            expected_fwd_output.append(test_case.model_forward_map(**mapped_input_i))
        expected_fwd_output = np.vstack(expected_fwd_output).T
        test_case.expected_fwd_output = cuqi.samples.Samples(expected_fwd_output, geometry=test_model.range_geometry)
        test_case.expected_fwd_output_type = cuqi.samples.Samples

        if test_model.gradient_form1 is not None:
            test_case.create_direction()
            test_case.expected_grad_output_value = NotImplementedError("Gradient is not implemented for input of type Samples.")

        test_model.test_data.append(test_case)
        
        # Case 7: inputs are samples but of different length # should raise an error
        test_case = TestCase(test_model)
        test_case.create_input()
        for i, (k, v) in enumerate(test_case.forward_input.items()):
            if i==0:
                v2 = 1.5*v
                samples = np.vstack([v, v2]).T
                test_case.forward_input[k] = cuqi.samples.Samples(samples, geometry=test_model.domain_geometry[i])
            else:
                v2 = 1.5*v
                v3 = 2*v
                samples = np.vstack([v, v2, v3]).T
                test_case.forward_input[k] = cuqi.samples.Samples(samples, geometry=test_model.domain_geometry[i])
        test_case.expected_fwd_output = ValueError(
                "If applying the function to Samples, all inputs should have the same number of samples.")

        if test_model.gradient_form1 is not None:
            test_case.create_direction()
            test_case.expected_grad_output_value = NotImplementedError("Gradient is not implemented for input of type Samples.")

        test_model.test_data.append(test_case)

model_test_case_combinations = TestMultipleInputModel.create_model_test_case_combinations()

def helper_function_for_printing_test_cases(model_test_case_combinations):
    """Helper function to print the test cases for debugging. This function
    is not used in the tests, but can be used by developers to print the
    test cases for debugging."""
    for i, (test_model, test_data) in enumerate(model_test_case_combinations):
        print("### Test Case {} ###".format(i))
        print("## test_model ##")
        print(test_model)
        print("## test_model._forward_func ##")
        print(test_model._forward_func)
        print("## test_model._gradient_func ##")
        print(test_model._gradient_func)
        print("## test_model._domain_geometry ##")
        print(test_model._domain_geometry)
        print("## test_model._range_geometry ##")
        print(test_model._range_geometry)
        print("="*50)
        print("## test_data ##")
        print(test_data)
        print("## test_data.forward_input ##")
        print(test_data.forward_input)
        print("## test_data.expected_fwd_output_type ##")
        print(test_data.expected_fwd_output_type)
        print("## test_data.expected_fwd_output ##")
        print(test_data.expected_fwd_output)
        print("## test_data.expected_grad_output_type ##")
        print(test_data.expected_grad_output_type)
        print("## test_data.expected_grad_output_value ##")
        print(test_data.expected_grad_output_value)
        print("#"*50)
        print("#"*50)

@pytest.mark.parametrize("test_model, test_data", model_test_case_combinations)
def test_multiple_input_model_forward(test_model, test_data):
    assert isinstance(test_model, cuqi.model.Model)
    assert isinstance(test_data, TestCase)
    if not isinstance(test_data.expected_fwd_output,
        (NotImplementedError, TypeError, ValueError)):
        fwd_output = test_model(**test_data.forward_input)
        fwd_output_stacked_inputs = test_model.forward(test_data.forward_input_stacked)
        assert len(test_data.forward_input_stacked) == test_model.domain_dim
        assert isinstance(fwd_output, test_data.expected_fwd_output_type)
        assert isinstance(fwd_output_stacked_inputs, np.ndarray)
        if isinstance(fwd_output, np.ndarray):
            assert np.allclose(fwd_output, test_data.expected_fwd_output)
            assert np.allclose(fwd_output_stacked_inputs, test_data.expected_fwd_output)
        elif isinstance(fwd_output, cuqi.samples.Samples):
            assert np.allclose(fwd_output.samples, test_data.expected_fwd_output.samples)
        else:
            raise NotImplementedError("Checks for other types of outputs not implemented.")
    else:
        with pytest.raises(type(test_data.expected_fwd_output), match=str(test_data.expected_fwd_output)):
            fwd_output = test_model(**test_data.forward_input)

@pytest.mark.parametrize("test_model, test_data", model_test_case_combinations)
def test_multiple_input_model_gradient(test_model, test_data):
    """Test that the gradient method can handle multiple inputs and
    return the correct output type and value"""

    assert isinstance(test_model, cuqi.model.Model)
    assert isinstance(test_data, TestCase)
    if hasattr(test_model,"_test_flag_gradient") and not test_model._test_flag_gradient:
        with pytest.raises(NotImplementedError, match="Gradient is not implemented for this model."):
            grad_output = test_model.gradient(test_data.direction, **test_data.forward_input)

    elif not isinstance(test_data.expected_grad_output_value,
        (NotImplementedError, TypeError, ValueError)):
        grad_output = test_model.gradient(test_data.direction, **test_data.forward_input)
        grad_output_stacked_inputs = test_model.gradient(test_data.direction, test_data.forward_input_stacked)

        # assert output format is a dictionary with keys x, y, z
        assert list(grad_output.keys()) == test_model._non_default_args
        assert list(grad_output_stacked_inputs.keys()) == test_model._non_default_args

        # Check type and value of the output (stacked)
        test_model._gradient_output_stacked = True
        stacked_grad_output = test_model.gradient(
            test_data.direction, **test_data.forward_input
        )
        test_model._gradient_output_stacked = False
        if np.all([value is not None for value in list(grad_output.values())]):
            expected_staked_grad_output = np.hstack(
                [
                    (
                        v.to_numpy()
                        if isinstance(v, CUQIarray)
                        else force_ndarray(v, flatten=True)
                    )
                    for v in list(test_data.expected_grad_output_value)
                ]
            )
            assert isinstance(stacked_grad_output, np.ndarray)
            assert np.allclose(stacked_grad_output, expected_staked_grad_output)

        # Check type and value of the output (not stacked)
        for i, (k, v) in enumerate(grad_output.items()):
            # Verify that the output is of the expected type
            if v is not None:
                assert isinstance(v, test_data.expected_grad_output_type)
                assert isinstance(grad_output_stacked_inputs[k], np.ndarray)

            # Verify that the output is of the expected value
            if isinstance(test_model._gradient_func, tuple) and test_model._gradient_func[i] is None:
                assert v is None
            else:
                assert np.allclose(v, test_data.expected_grad_output_value[i])
                assert np.allclose(v, grad_output_stacked_inputs[k])
                assert np.allclose(v, test_data.FD_grad_output[i])
    else:
        with pytest.raises(type(test_data.expected_grad_output_value), match=str(test_data.expected_grad_output_value)):
            grad_output = test_model.gradient(test_data.direction, **test_data.forward_input)

@pytest.mark.parametrize("seed",[(0),(1),(2)])
def test_LinearModel_getMatrix(seed):
    np.random.seed(seed)
    A = np.random.randn(10,7) #Random matrix

    model1 = cuqi.model.LinearModel(A)
    model2 = cuqi.model.LinearModel(lambda x : A@x, lambda y: A.T@y, range_geometry=A.shape[0], domain_geometry=A.shape[1])
    
    mat1 = model1.get_matrix() #Normal matrix
    mat2 = model2.get_matrix() #Sparse matrix (generated from functions)

    assert np.allclose(mat1,mat2.A)

def test_initialize_model_dim():
    model1 = cuqi.model.Model(lambda x:x, range_geometry=4, domain_geometry=4)
    assert(len(model1.domain_geometry.grid) == 4 and len(model1.range_geometry.grid) == 4 )

def test_initialize_model_continuous1D_geom():
    range_geometry = cuqi.geometry.Continuous1D(grid=5)
    domain_geometry = cuqi.geometry.Continuous1D(grid=3)
    model1 = cuqi.model.Model(lambda x:x,range_geometry, domain_geometry)
    dims_old = (model1.range_dim, model1.domain_dim) 
    model1.range_geometry = cuqi.geometry.Continuous1D(grid=4)
    assert(dims_old == (5,3) and (model1.range_dim, model1.domain_dim) == (4,3))

def test_initialize_model_continuous2D_geom():
    range_geometry = cuqi.geometry.Continuous2D(grid=([1,2,3,4,5,6],4))
    domain_geometry = cuqi.geometry.Continuous2D(grid=(np.array([0,.5,1,2]),[1,2,3,4]))
    model1 = cuqi.model.Model(lambda x:x,range_geometry, domain_geometry)
    dims_old = (model1.range_dim, model1.domain_dim) 
    model1.range_geometry = cuqi.geometry.Continuous2D(grid=(10,4))
    assert(dims_old == (24,16) and (model1.range_dim, model1.domain_dim) == (40,16)) 

def test_initialize_model_matr():
    model1 = cuqi.model.LinearModel(np.eye(5))
    assert( (model1.range_dim, model1.domain_dim) == (5,5) and model1.domain_geometry.par_shape == (5,) and
            len(model1.range_geometry.grid) == 5)

def test_model_allow_DefaultGeometry():
    """ Tests that model can have specific geometry (Image2D) and x can be _Defaultgeometry"""
    model = cuqi.testproblem.Deconvolution2D(dim=5).model
    x = cuqi.distribution.Gaussian(np.zeros(model.domain_dim), 1).sample()
    model(x)   #Forward
    model.T(x) #Adjoint

@pytest.mark.parametrize("x, expected_type",
                         [(np.array([1, 3, 4]),
                           np.ndarray),

                          (cuqi.samples.Samples(
                              samples=np.array([[1, 3, 4],
                                               [5, 2, 6]]).T),
                           cuqi.samples.Samples),

                          (cuqi.array.CUQIarray(np.array([1, 3, 4]),
                            geometry=cuqi.geometry.Continuous1D(3)),
                           cuqi.array.CUQIarray)])
def test_forward(x, expected_type):
    """For different types of input to the model forward method, assert we are obtaining the correct output type"""
    A = np.array([[1, 0, 0],[0, 3, .1]])

    model = cuqi.model.Model(forward=lambda x:A@x,
                            gradient=lambda direction,
                            x: A.T@direction,
                            domain_geometry=cuqi.geometry.Continuous1D(3),
                            range_geometry=cuqi.geometry.Continuous1D(2))

    fwd = model.forward(x)
    assert(isinstance(fwd, expected_type))


@pytest.mark.parametrize(
    "x, y, z, forward, domain_geometry, range_geometry, expected_value, expected_type",
    [
        (
            np.array([1, 3, 4]),
            np.array([1, 3]),
            3,
            lambda x, y, z: x * y[0] + z * y[1],
            cuqi.experimental.geometry._ProductGeometry(
                cuqi.geometry.Continuous1D(3),
                cuqi.geometry.Continuous1D(2),
                cuqi.geometry.Continuous1D(3),
            ),
            cuqi.geometry.Continuous1D(3),
            np.array([10, 12, 13]),
            np.ndarray,
        )
    ],
)
class TestForwardWithMultipleInputs:
    def test_forward_with_multiple_inputs(
        self,
        x,
        y,
        z,
        forward,
        domain_geometry,
        range_geometry,
        expected_value,
        expected_type,
    ):
        """Test that the forward method can handle multiple inputs and
        return the correct output type"""

        model = cuqi.model.Model(
            forward=forward,
            domain_geometry=domain_geometry,
            range_geometry=range_geometry,
        )

        fwd = model.forward(x, y, z)
        fwd2 = model(y=y, x=x, z=z)
        fwd3 = model.forward(x, y, z, is_par=True)
        fwd4 = model(z=z, x=x, y=y, is_par=True)
        assert isinstance(fwd, expected_type)
        assert np.allclose(fwd, expected_value)
        assert np.allclose(fwd, fwd2)
        assert np.allclose(fwd, fwd3)
        assert np.allclose(fwd, fwd4)


    def test_forward_with_multiple_inputs_error_when_mixing_args_and_kwargs(
        self,
        x,
        y,
        z,
        forward,
        domain_geometry,
        range_geometry,
        expected_value,
        expected_type,
    ):
        """Test that the forward method raises an error when mixing positional
        arguments and keyword arguments"""

        model = cuqi.model.Model(
            forward=forward,
            domain_geometry=domain_geometry,
            range_geometry=range_geometry,
        )

        with pytest.raises(
            ValueError,
            match=r"The model input is specified both as positional and keyword arguments. This is not supported",
        ):
            fwd = model.forward(x, y=y, z=z)


@pytest.mark.parametrize("x, expected_type",
                         [(np.array([1, 3]),
                           np.ndarray),

                          (cuqi.samples.Samples(
                              samples=np.array([[1, 3],
                                               [5, 2]])),                
                           cuqi.samples.Samples),

                          (cuqi.array.CUQIarray(np.array([1, 3]),
                            geometry=cuqi.geometry.Continuous1D(2)),
                           cuqi.array.CUQIarray)])
def test_adjoint(x, expected_type):
    """For different types of input to the model adjoint method, assert we are obtaining the correct output type"""
    A = np.array([[1, 0, 0],[0, 3, .1]])

    model = cuqi.model.LinearModel(forward=A,
                                   domain_geometry=cuqi.geometry.Continuous1D(3),
                                   range_geometry=cuqi.geometry.Continuous1D(2))

    ad = model.adjoint(x)
    assert(isinstance(ad, expected_type))

@pytest.mark.parametrize("direction, expected_type",
                         [(np.array([1, 4]),
                           np.ndarray),

                          (cuqi.samples.Samples(
                              samples=np.array([[3, 4],
                                               [2, 6]]).T),
                           cuqi.samples.Samples),

                          (cuqi.array.CUQIarray(np.array([3, 4]),
                            geometry=cuqi.geometry.Continuous1D(2)),
                           cuqi.array.CUQIarray)])
def test_gradient(direction, expected_type):
    """For different types of input to the model gradient method, assert we are obtaining the correct output type"""
    A = np.array([[1, 0, 0],[0, 3, .1]])

    model = cuqi.model.Model(forward=lambda x:A@x,
                            gradient=lambda direction,
                            x: A.T@direction,
                            domain_geometry=cuqi.geometry.Continuous1D(3),
                            range_geometry=cuqi.geometry.Continuous1D(2))

    if isinstance(direction, cuqi.samples.Samples):
        with pytest.raises(NotImplementedError, match="Gradient is not implemented for input of type Samples."):
           grad = model.gradient(direction, None)
    else:            
        grad = model.gradient(direction, None)
        assert(isinstance(grad, expected_type))


@pytest.mark.parametrize("wrt, is_wrt_par, case_id",
                         [(np.array([1, 4]), False, 1),
                          (np.array([1, 4]), True, 2),
                          (np.array([1, 4]), False, 3),
                          (np.array([1, 4]), True, 4)])
def test_gradient_raised_errors(wrt, is_wrt_par, case_id):
    """ Test different types of wrt input"""
    range_geometry = cuqi.geometry.Continuous1D(2)
    if case_id == 1:
        domain_geometry = \
         cuqi.geometry.MappedGeometry(
             cuqi.geometry.Continuous1D(2), map=lambda x:x)
    if case_id == 2:
        domain_geometry = \
         cuqi.geometry.MappedGeometry(
             cuqi.geometry.Continuous1D(2), map=lambda x:2*x,
             imap= lambda x:x/2)

    if case_id == 3:
        domain_geometry = \
         cuqi.geometry.Continuous1D(2)
        def my_fun2par(p):
            raise NotImplementedError 
        domain_geometry.fun2par = my_fun2par

    if case_id == 4:
        domain_geometry = \
         cuqi.geometry.Continuous1D(2)
        range_geometry = \
         cuqi.geometry.MappedGeometry(
             cuqi.geometry.Continuous1D(2), map=lambda x:2*x,
             imap= lambda x:x/2)

    A = np.array([[1, 0],[0, 3]])
    direction = np.array([1, 4])

    model = cuqi.model.Model(forward=lambda x:x@A@x,
                            gradient=lambda direction,
                            x: 2*x@A@direction,
                            domain_geometry=domain_geometry,
                            range_geometry=range_geometry)

    if case_id ==1:
        # Raises and error because imap is not implemented for
        # the domain_geometry and hence can't compute fun2par 
        # (and wrt is passed as function).
        with pytest.raises(ValueError):
            grad = model.gradient(direction, wrt, is_var_par=is_wrt_par)

    if case_id ==2:
        # Raises an error because gradient is not implemented for
        # the domain_geometry and the domain geometry is not in
        # the list cuqi.geometry._get_identity_geometries()
        with pytest.raises(NotImplementedError):
            grad = model.gradient(direction, wrt, is_var_par=is_wrt_par)

    if case_id ==3:
        # Raises an error because domain_geometry does not have an
        # implementation of fun2par and wrt is passed as function.
        with pytest.raises(NotImplementedError):
            grad = model.gradient(direction, wrt, is_var_par=is_wrt_par)

    if case_id == 4:
        # Raises an error because the range_geometry is not in the
        # cuqi.geometry._get_identity_geometries() list
        with pytest.raises(NotImplementedError):
            grad = model.gradient(direction, wrt, is_var_par=is_wrt_par)


@pytest.mark.parametrize("forward, gradient, direction, x, domain_geometry, domain_gradient, range_geometry, is_direction_par, is_wrt_par",
                         [
                             (
                                 lambda x: np.array([[1, 0, 0], [0, 3, .1]])@x,
                                 lambda direction, x: np.array(
                                     [[1, 0, 0], [0, 3, .1]]).T@direction,
                                 np.array([1, 12]),
                                 np.array([1, 12, 8]),
                                 cuqi.geometry.MappedGeometry(Continuous1D(
                                     3), map=lambda x:2*x, imap=lambda x:x/2),
                                 lambda direction, wrt:2*np.eye(3)@direction,
                                 cuqi.geometry.Continuous1D(2),
                                 True,
                                 True
                             ),
                             (
                                 lambda x: np.array([[1, 2, 8], [1, 3, .1]])@x,
                                 lambda direction, x: np.array(
                                     [[1, 2, 8], [1, 3, .1]]).T@direction,
                                 np.array([1, 12]),
                                 np.array([1, 12, 8]),
                                 cuqi.geometry.MappedGeometry(Continuous1D(
                                     3), map=lambda x:x**2, imap=lambda x:np.sqrt(x)),
                                 lambda direction, wrt:2 *
                                 np.diag(wrt)@direction,
                                 cuqi.geometry.Continuous1D(2),
                                 True,
                                 True
                             ),
                             (
                                 lambda x: np.sin(x),
                                 lambda direction, x: np.diag(
                                     np.cos(x))@direction,
                                 np.array([1, 1, 4]),
                                 np.array([1, 12, 8]),
                                 cuqi.geometry.MappedGeometry(Continuous1D(
                                     3), map=lambda x:x**2, imap=lambda x:np.sqrt(x)),
                                 lambda direction, x:2*np.diag(x)@direction,
                                 cuqi.geometry.Continuous1D(3),
                                 True,
                                 True
                             ),
                             (
                                 lambda x: np.sin(x),
                                 lambda direction, x: np.diag(
                                     np.cos(x))@direction,
                                 np.array([1, 1, 4]),
                                 cuqi.array.CUQIarray(
                                     np.array([1, 12**2, 8**2]), is_par=False, geometry=Continuous1D(3)), # Geometry will be updated
                                 cuqi.geometry.MappedGeometry(Continuous1D(
                                     3), map=lambda x:x**2, imap=lambda x:np.sqrt(x)),
                                 lambda direction, x:2*np.diag(x)@direction,
                                 cuqi.geometry.Continuous1D(3),
                                 True,
                                 True
                             ),
                             (
                                 lambda x: np.sin(x),
                                 lambda direction, x: np.diag(
                                     np.cos(x))@direction,
                                 np.array([1, 1, 4]),
                                 np.array([1, 12**2, 8**2]),
                                 cuqi.geometry.MappedGeometry(Continuous1D(
                                     3), map=lambda x:x**2, imap=lambda x:np.sqrt(x)),
                                 lambda direction, x:2*np.diag(x)@direction,
                                 cuqi.geometry.Continuous1D(3),
                                 True,
                                 False
                             ),
                             (
                                 lambda x: np.sin(x),
                                 lambda direction, x: np.diag(
                                     np.cos(x))@direction,
                                 cuqi.array.CUQIarray(
                                     np.array([1, 1, 4]), is_par=False, geometry=Continuous1D(3)), # Geometry will be updated
                                 np.array([1, 12, 8]),
                                 cuqi.geometry.MappedGeometry(Continuous1D(
                                     3), map=lambda x:x**2, imap=lambda x:np.sqrt(x)),
                                 lambda direction, x:2*np.diag(x)@direction,
                                 cuqi.geometry.Continuous1D(3),
                                 True,
                                 True
                             ),
                             (
                                 lambda x: np.sin(x),
                                 lambda direction, x: np.diag(
                                     np.cos(x))@direction,
                                 np.array([1, 1, 4]),
                                 np.array([1, 12, 8]),
                                 cuqi.geometry.MappedGeometry(Continuous1D(
                                     3), map=lambda x:x**2, imap=lambda x:np.sqrt(x)),
                                 lambda direction, x:2*np.diag(x)@direction,
                                 cuqi.geometry.Continuous1D(3),
                                 False,
                                 True
                             ),
                             (
                                 lambda x: np.sin(x),
                                 lambda direction, x: np.diag(
                                     np.cos(x))@direction,
                                 np.array([1, 1, 4]),
                                 np.array([1, 12**2, 8**2]),
                                 cuqi.geometry.MappedGeometry(Continuous1D(
                                     3), map=lambda x:x**2, imap=lambda x:np.sqrt(x)),
                                 lambda direction, x:2*np.diag(x)@direction,
                                 cuqi.geometry.Continuous1D(3),
                                 False,
                                 False
                             ),
                         ]
                         )
def test_gradient_computation(forward, gradient, direction, x, domain_geometry, domain_gradient, range_geometry, is_direction_par, is_wrt_par):
    """Test applying chain rule when computing the gradient"""
    model = cuqi.model.Model(forward=forward,
                             gradient=gradient,
                             domain_geometry=domain_geometry,
                             range_geometry=range_geometry)
    model.domain_geometry.gradient = domain_gradient

    # Set CUQIarray geometries
    if isinstance(x, CUQIarray):
        x.geometry = domain_geometry
    if isinstance(direction, CUQIarray):
        direction.geometry = range_geometry
    # Compute cuqi model gradient 
    grad = model.gradient(direction, x, is_var_par=is_wrt_par)

    # If wrt is function value, convert wrt to parameters to
    # be passed as an input for to cuqi.utilities.approx_derivative
    if isinstance(x, CUQIarray):
        wrt_ndarray = np.zeros(len(x.parameters))
        wrt_ndarray[:] = x.parameters
        x = wrt_ndarray
    if not is_wrt_par:
        x = domain_geometry.fun2par(x)

    if isinstance(direction, CUQIarray):
        direction_ndarray = np.zeros(len(direction.parameters))
        direction_ndarray[:] = direction.parameters
        direction = direction_ndarray
    if not is_direction_par:
        direction = range_geometry.fun2par(direction)
    # Compute a finite difference approximation of cuqi model gradient 
    findiff_grad = cuqi.utilities.approx_derivative(
        model.forward, x, direction)

    # Compare the two gradients
    assert(np.allclose(grad, findiff_grad))

def test_model_parameter_name_switch():
    """ Test that the model can switch its parameter name if given a distribution as input """

    model = cuqi.testproblem.Deconvolution1D().model 

    # Parameter name defaults to 'x'
    assert cuqi.utilities.get_non_default_args(model) == ['x']

    # Parameter name can be switched to 'y' by "evaluating" the model with y as input
    y = cuqi.distribution.Gaussian(np.zeros(model.domain_dim), 1)
    model_y = model(y)
    model_y2 = model@y
    model_y3 = model.forward(y)

    # Check that the model has switched its parameter name
    assert cuqi.utilities.get_non_default_args(model_y) == ['y']
    assert cuqi.utilities.get_non_default_args(model_y2) == ['y']
    assert cuqi.utilities.get_non_default_args(model_y3) == ['y']

def test_model_parameter_name_switch_errors():
    """ Check that an error is raised if dim does not match or if distribution has no name """

    model = cuqi.testproblem.Poisson1D().model # domain_dim != range_dim

    y = cuqi.distribution.Gaussian(np.zeros(model.domain_dim-1), 1)

    with pytest.raises(ValueError, match=r"Attempting to match parameter name of Model with given distribution\(s\), but distribution\(s\) dimension\(s\) does not match model input dimension\(s\)."):
        model(y)    

def test_model_allow_other_parameter_names():
    """ Test that Model correctly infers parameter names from the forward function and evaluation matches. """

    forward_x = lambda x: x
    forward_y = lambda y: y
    def forward_z(z, not_used=None):
        return z

    model_x = cuqi.model.Model(forward_x, 1, 1)
    model_y = cuqi.model.Model(forward_y, 1, 1)
    model_z = cuqi.model.Model(forward_z, 1, 1)

    # Check that the model has switched its parameter name
    assert model_x._non_default_args == ['x']
    assert model_y._non_default_args == ['y']
    assert model_z._non_default_args == ['z']

    # Check that we can provide parameter names when evaluating the model
    assert model_x(x=1) == 1
    assert model_y(y=1) == 1
    assert model_z(z=1) == 1

    # And check that we can provide positional arguments
    assert model_x(1) == 1
    assert model_y(1) == 1
    assert model_z(1) == 1

def test_linear_model_allow_other_parameter_names():
    """ Test that linear model automatically infers parameter names from the forward function and evaluation matches. """

    forward_x = lambda x: x
    forward_y = lambda y: y
    def forward_z(z, not_used=None):
        return z

    adjoint = lambda w: w

    model_x = cuqi.model.LinearModel(forward_x, adjoint, 1, 1)
    model_y = cuqi.model.LinearModel(forward_y, adjoint, 1, 1)
    model_z = cuqi.model.LinearModel(forward_z, adjoint, 1, 1)

    A = np.array([[1]])
    model_mat = cuqi.model.LinearModel(A) # Default parameter name is 'x'

    # Check that the model has switched its parameter name
    assert model_x._non_default_args == ['x']
    assert model_y._non_default_args == ['y']
    assert model_z._non_default_args == ['z']
    assert model_mat._non_default_args == ['x']

    # Check that we can provide parameter names when evaluating the model
    assert model_x(x=1) == 1
    assert model_y(y=1) == 1
    assert model_z(z=1) == 1
    assert model_mat(x=np.ones(1)) == np.ones(1) # With matrix it has to be numpy array

    # And check that we can provide positional arguments
    assert model_x(1) == 1
    assert model_y(1) == 1
    assert model_z(1) == 1
    assert model_mat(np.ones(1)) == np.ones(1)

    # Check that matrix multiplication works
    assert model_x@1 == 1
    assert model_y@1 == 1
    assert model_z@1 == 1
    assert model_mat@np.ones(1) == np.ones(1)

def test_model_allows_jacobian_or_gradient():
    """ Test that either Jacobian or gradient (vector-jacobian product) can be specified and that it gives the same result. """
    # This is the Wang Cubic test problem model
    def forward(x):
        return 10*x[1] - 10*x[0]**3 + 5*x[0]**2 + 6*x[0]
    def jacobian(x):
        return np.array([[-30*x[0]**2 + 10*x[0] + 6, 10]])
    def gradient(direction, x):
        return direction@jacobian(x)
    
    # Check not both can be specified
    with pytest.raises(TypeError, match=r"Only one of gradient and jacobian"):
        model = cuqi.model.Model(forward, range_geometry=1, domain_geometry=2, jacobian=jacobian, gradient=gradient)

    # Check that we can specify jacobian
    model_jac = cuqi.model.Model(forward, range_geometry=1, domain_geometry=2, jacobian=jacobian)

    # Check that we can specify gradient
    model_grad = cuqi.model.Model(forward, range_geometry=1, domain_geometry=2, gradient=gradient)

    # Check that we can evaluate the model gradient and get the same result as the jacobian
    wrt = np.random.randn(1)
    dir = np.random.randn(1)

    assert np.allclose(model_grad.gradient(dir, wrt), model_jac.gradient(dir, wrt))

# Parametrize over models
@pytest.mark.parametrize("model", [cuqi.testproblem.Deconvolution1D().model])
def test_AffineModel_Correct_result(model):
    """ Test creating a shifted linear model from a linear model """

    # Random vectors
    x = np.random.randn(model.domain_dim)
    b = np.random.randn(model.range_dim)

    A_affine = cuqi.model.AffineModel(model, b, range_geometry = model.range_geometry, domain_geometry = model.domain_geometry)

    # Dimension check
    assert A_affine.range_dim == model.range_dim

    # Check that the shifted linear model is correct
    assert np.allclose(A_affine(x), model(x) + b)

@pytest.mark.parametrize("model", [cuqi.testproblem.Heat1D().model])
def test_AffineModel_does_not_accept_general_model(model):
    """ Test that AffineModel only accepts LinearModel, besides accepting
    a matrix or callable"""

    # Random vectors
    x = np.random.randn(model.domain_dim)
    b = np.random.randn(model.range_dim)
    
    with pytest.raises(TypeError,
                       match=r"The linear operator should be a LinearModel object, a callable function or a matrix."):
        A_affine = cuqi.model.AffineModel(model, b, range_geometry = model.range_geometry, domain_geometry = model.domain_geometry)

@pytest.mark.parametrize("fwd_func", [lambda: 1, lambda x, y: x])
def test_AffineModel_raise_error_when_forward_has_too_few_or_too_many_inputs(
    fwd_func
):
    """ Test that AffineModel raises an error when the forward function has too
    few or too many inputs """
    adj_func = lambda x: x
    with pytest.raises(ValueError,
                       match=r"The linear operator should have exactly one input argument"):
        A_affine = cuqi.model.AffineModel(fwd_func, 0, adj_func)
    
    # reverse passing the forward and adjoint functions
    with pytest.raises(ValueError,
                       match=r"The adjoint linear operator should have exactly one input argument"):
        A_affine = cuqi.model.AffineModel(adj_func, 0, fwd_func)

def test_AffineModel_update_shift():

    A = np.eye(2)
    b = np.array([1, 2])
    x = np.array([1, 1])
    new_shift = np.array([2,-1])
    model = cuqi.model.AffineModel(A, b)
    model_copy = copy(model)

    # check model output
    assert np.all(model(x) == np.array([2,3]))

    # check model output with updated shift
    model.shift = new_shift
    assert np.all(model(x) == np.array([3,0]))

    # check model output of copied model
    assert np.all(model_copy(x) == np.array([2,3]))

    # check model output of copied model with updated shift
    model_copy.shift = new_shift
    assert np.all(model_copy(x) == np.array([3,0]))

def test_PDE_model_multiple_input():
    """ Test that the PDE model can handle multiple inputs and return the correct output type"""
    pde_test_model = TestMultipleInputModel.helper_build_steady_state_PDE_test_model()
    pde_test_model.populate_model_variations()
    CUQI_pde = pde_test_model.model_variations[0] # PDE model with multiple inputs

    # Check that the model has switched its parameter name
    assert CUQI_pde._non_default_args == ['mag', 'kappa_scale']

    # Check that we can provide parameter names when evaluating the model
    assert CUQI_pde(mag=2, kappa_scale=2) is not None

    # And check that we can provide positional arguments
    assert CUQI_pde(2, 2) is not None

def test_constructing_gradient_from_jacobian():
    """ Test that the gradient is correctly constructed from the
    jacobian when only the jacobian is specified """

    model = cuqi.model.Model(
        lambda x, y: x * 2 + y,
        domain_geometry=(Discrete(1), Discrete(1)),
        range_geometry=1,
        jacobian=lambda x, y: np.array([[2, 1]]),
    )

    # Evaluate the gradient which was constructed from the jacobian:
    grad1 = model.gradient(x=1, y=1, direction=np.array([2]))
    # stack grad1
    grad1 = np.hstack([grad1[k] for k in grad1])

    # Compute the gradient using the FD approximation:

    # slightly different form of the forward function
    # that takes a single input (which approx_derivative requires)
    forward2 = lambda x: x[0] * 2 + x[1]
    grad2 = cuqi.utilities.approx_derivative(
        forward2, np.array([1, 1]), np.array([2]), 1e-6
    )

    assert np.allclose(grad1, grad2)


def test_model_updates_parameters_names_if_distributions_are_passed_with_new_parameter_names():
    """ Test that the model changes the parameter names if given a distribution as input with new parameter names """

    def forward(x, y):
        return y*x[0] + x[1]

    def gradient_x(direction, x, y):
        return direction*np.array([y, 1])

    def gradient_y(direction, x, y):
        return direction*x[0]

    # forward model inputs
    input1 = np.array([1, 2])
    input2 = 1
    direction = 3

    model = cuqi.model.Model(forward=forward, range_geometry=1, domain_geometry=(Continuous1D(2), Discrete(1)), gradient=(gradient_x, gradient_y))

    # assert model parameter names are 'x' and 'y'
    assert model._non_default_args == ['x', 'y']

    # create two random distributions a and b
    a = cuqi.distribution.Gaussian(np.zeros(2), 1)
    b = cuqi.distribution.Gaussian(np.zeros(1), 1)

    model_a_b = model(a, b)

    # assert model still has parameter names 'x' and 'y'
    assert model._non_default_args == ['x', 'y']

    # assert new model parameter names are 'a' and 'b'
    assert model_a_b._non_default_args == ['a', 'b']

    # assert the two models output are equal
    model_output = model(x=input1, y=input2)
    model_a_b_output = model_a_b(b=input2, a=input1)
    assert np.allclose(model_output, model_a_b_output)

    # assert the two models gradient are equal
    model_grad_v1 = list(model.gradient(direction, x=input1, y=input2).values())
    model_grad_v2 = list(model.gradient(direction, input1, input2).values())
    model_grad_v3 = list(model.gradient(direction, y=input2, x=input1).values())
    model_a_b_grad_v1 = list(model_a_b.gradient(direction, a=input1, b=input2).values())
    model_a_b_grad_v2 = list(model_a_b.gradient(direction, input1, input2).values())
    model_a_b_grad_v3 = list(model_a_b.gradient(direction, b=input2, a=input1).values())

    # assert all gradients are equal
    for i in range(len(model_grad_v1)):
        assert np.allclose(model_grad_v1[i], model_grad_v2[i])
        assert np.allclose(model_grad_v1[i], model_grad_v3[i])
        assert np.allclose(model_grad_v1[i], model_a_b_grad_v1[i])
        assert np.allclose(model_grad_v1[i], model_a_b_grad_v2[i])
        assert np.allclose(model_grad_v1[i], model_a_b_grad_v3[i])

    # compare to FD gradient
    grad_FD = cuqi.utilities.approx_derivative(
        lambda par: model.forward(par[:2], par[-1]),
        np.hstack([input1, np.array([input2])]),
        np.array([direction]))
    assert np.allclose(model_grad_v1[0], grad_FD[:2])
    assert np.allclose(model_grad_v1[1], grad_FD[-1])

def test_linear_model_updates_parameters_names_if_distributions_are_passed_with_new_parameter_names():
    """ Test that the linear model changes parameter names if given a distribution as input with new parameter names """

    model = cuqi.testproblem.Deconvolution1D().model
    model_input = np.random.randn(model.domain_dim)

    # assert model parameter names are 'x'
    assert model._non_default_args == ['x']

    # create a random distribution a
    a = cuqi.distribution.Gaussian(np.zeros(model.domain_dim), 1)

    model_a = model(a)

    # assert model still has parameter names 'x'
    assert model._non_default_args == ['x']
    # assert new model parameter names are 'a'
    assert model_a._non_default_args == ['a']

    # assert the two models output are equal
    assert np.allclose(model(x=model_input), model_a(a=model_input))

    # assert the two models gradient are equal
    direction = np.random.randn(model.range_dim)
    model_grad = model.gradient(direction, x=model_input)
    model_a_grad = model_a.gradient(direction, a=model_input)
    assert np.allclose(model_grad, model_a_grad)

    # assert the two models adjoint are equal
    model_adjoint = model.adjoint(y=model_input)
    model_a_adjoint = model_a.adjoint(y=model_input)
    assert np.allclose(model_adjoint, model_a_adjoint)
