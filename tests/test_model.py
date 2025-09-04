from cuqi.array import CUQIarray
from cuqi.samples import Samples
from cuqi.geometry import Continuous1D, Discrete
import numpy as np
import scipy as sp
import cuqi
import pytest
from scipy import optimize
from copy import copy, deepcopy
from cuqi.geometry import _identity_geometries, _DefaultGeometry1D, _DefaultGeometry2D, Geometry, Discrete, Image2D, KLExpansion
from cuqi.utilities import force_ndarray
from cuqi.experimental.geometry import _ProductGeometry

def test_steady_state_PDE_model_multiple_input():
    """ Test that the steady state PDE model and gradient can accept multiple inputs specified as positional arguments or keyword arguments """
    pde_test_model = MultipleInputTestModel.helper_build_steady_state_PDE_test_model()
    pde_test_model.populate_model_variations()
    CUQI_pde = pde_test_model.model_variations[1] # PDE model with multiple inputs

    # Check that the model has correct parameter name
    assert CUQI_pde._non_default_args == ['mag', 'kappa_scale']

    # Check that we can provide parameter names when evaluating the model
    output1 = CUQI_pde(mag=2, kappa_scale=2)

    # And check that we can provide positional arguments
    output2 = CUQI_pde(2, 2)

    # Check that the two outputs are the same
    assert np.allclose(output1, output2)

    # Assert evaluating gradient works
    direction = np.random.randn(CUQI_pde.range_dim)

    # Make sure gradient can be computed with positional or keyword arguments
    grad1 = CUQI_pde.gradient(direction, mag=2, kappa_scale=2)
    grad2 = CUQI_pde.gradient(direction, 2, 2)

    # Passing wrong kwargs should raise an error
    with pytest.raises(
        ValueError,
        match=r"The gradient input is specified by a direction and keywords arguments \['mag', 'kappa'\] that does not match the non_default_args of the model \['mag', 'kappa_scale'\].",
    ):
        CUQI_pde.gradient(direction, mag=2, kappa=2)

def test_time_dependent_PDE_model_multiple_input():
    """ Test that the time dependent PDE model and gradient can accept multiple inputs specified as positional arguments or keyword arguments """
    pde_test_model = MultipleInputTestModel.helper_build_time_dependent_PDE_test_model()
    pde_test_model.populate_model_variations()
    CUQI_pde = pde_test_model.model_variations[1] # PDE model with multiple inputs

    # Check that the model has correct parameter name
    assert CUQI_pde._non_default_args == ['mag', 'IC']

    # Check that we can provide parameter names when evaluating the model
    mag = 2
    IC_ = np.random.randn(CUQI_pde.domain_geometry.geometries[1].par_dim)
    output1 = CUQI_pde(mag=mag, IC=IC_)

    # And check that we can provide positional arguments
    output2 = CUQI_pde(mag, IC_)

    # Check that the two outputs are the same
    assert np.allclose(output1, output2)

    # Assert evaluating gradient works
    direction = np.random.randn(CUQI_pde.range_dim)

    # Make sure gradient can be computed with positional or keyword arguments
    grad1 = CUQI_pde.gradient(direction, mag=mag, IC=IC_)
    grad2 = CUQI_pde.gradient(direction, mag, IC_)

    # Passing wrong kwargs should raise an error
    with pytest.raises(
        ValueError,
        match=r"The gradient input is specified by a direction and keywords arguments \['mag', 'IC_value'\] that does not match the non_default_args of the model \['mag', 'IC'\].",
    ):
        CUQI_pde.gradient(direction, mag=mag, IC_value=IC_)

def test_constructing_gradient_from_jacobian():
    """ Test that the gradient is correctly constructed from the
    jacobian when only the jacobian is specified """

    model = cuqi.model.Model(
        lambda x, y: x * 2 + y,
        domain_geometry=(Discrete(1), Discrete(1)),
        range_geometry=1,
        jacobian=lambda x, y: (np.array([2]), np.array([1])),
    )

    # Evaluate the gradient which was constructed from the jacobian:
    grad1 = model.gradient(x=1, y=1, direction=np.array([2]))
    # Stack grad1
    grad1 = np.hstack([grad1[k] for k in grad1])

    # Compute the gradient using the FD approximation:

    # Slightly different form of the forward function
    # that takes a single input (which approx_derivative requires)
    forward2 = lambda x: x[0] * 2 + x[1]
    grad2 = cuqi.utilities.approx_derivative(
        forward2, np.array([1, 1]), np.array([2]), 1e-6
    )

    assert np.allclose(grad1, grad2)

@pytest.mark.parametrize(
    "obj_type", ['distribution', 'random_variable'])
def test_model_updates_parameters_names_to_follow_distribution_or_random_variable_names(obj_type):
    """Test that the model changes the parameter names if given distributions or random variables as input with new parameter names"""

    def forward(x, y):
        return y * x[0] + x[1]

    def gradient_x(direction, x, y):
        return direction * np.array([y, 1])

    def gradient_y(direction, x, y):
        return direction * x[0]

    # forward model inputs
    input1 = np.array([1, 2])
    input2 = 1
    direction = 3

    model = cuqi.model.Model(
        forward=forward,
        range_geometry=1,
        domain_geometry=(Continuous1D(2), Discrete(1)),
        gradient=(gradient_x, gradient_y),
    )

    # assert model parameter names are 'x' and 'y'
    assert model._non_default_args == ["x", "y"]
    
    if obj_type == 'distribution':
        # create two random distributions a and b
        a = cuqi.distribution.Gaussian(np.zeros(2), 1)
        b = cuqi.distribution.Gaussian(np.zeros(1), 1)
    elif obj_type == 'random_variable':
        # create two random variables a and b
        a = cuqi.distribution.Gaussian(np.zeros(2), 1).rv
        b = cuqi.distribution.Gaussian(np.zeros(1), 1).rv
    else:
        raise ValueError("obj_type must be either 'distribution' or 'random_variable'")

    model_a_b = model(a, b)

    # assert model still has parameter names 'x' and 'y'
    assert model._non_default_args == ["x", "y"]

    # assert new model parameter names are 'a' and 'b'
    assert model_a_b._non_default_args == ["a", "b"]

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
        np.array([direction]),
    )
    assert np.allclose(model_grad_v1[0], grad_FD[:2])
    assert np.allclose(model_grad_v1[1], grad_FD[-1])

@pytest.mark.parametrize(
    "model",
    [
        cuqi.testproblem.Deconvolution1D().model,
        cuqi.model.LinearModel(lambda x: x, lambda y: y, 1, 1),
    ],
)
@pytest.mark.parametrize("obj_type", ['distribution', 'random_variable'])
def test_linear_model_updates_parameters_names_to_follow_distribution_or_random_variable_names(
    model, obj_type
):
    """Test that the linear model changes parameter names if given a distribution or a random variable as input with new parameter names"""

    model_input = np.random.randn(model.domain_dim)

    # assert model parameter names are 'x'
    assert model._non_default_args == ["x"]
    
    if obj_type == 'distribution':
        # create a random distribution a
        a = cuqi.distribution.Gaussian(np.zeros(model.domain_dim), 1)
    elif obj_type == 'random_variable':
        # create a random variable a
        a = cuqi.distribution.Gaussian(np.zeros(model.domain_dim), 1).rv
    else:
        raise ValueError("obj_type must be either 'distribution' or 'random_variable'")

    model_a = model(a)

    # assert model still has parameter names 'x'
    assert model._non_default_args == ["x"]
    # assert new model parameter names are 'a'
    assert model_a._non_default_args == ["a"]

    # assert the two models output are equal
    assert np.allclose(model(x=model_input), model_a(a=model_input))
    # assert _forward_func_no_shift also works with new parameter names
    assert np.allclose(
        model._forward_func_no_shift(x=model_input),
        model_a._forward_func_no_shift(a=model_input),
    )
    # and also when inputs are positional arguments
    assert np.allclose(
        model._forward_func_no_shift(model_input),
        model_a._forward_func_no_shift(model_input),
    )

    # assert the two models gradient are equal
    direction = np.random.randn(model.range_dim)
    model_grad = model.gradient(direction, x=model_input)
    model_a_grad = model_a.gradient(direction, a=model_input)
    assert np.allclose(model_grad, model_a_grad)

    # assert the two models adjoint are equal
    model_adjoint = model.adjoint(y=model_input)
    model_a_adjoint = model_a.adjoint(y=model_input)
    assert np.allclose(model_adjoint, model_a_adjoint)

@pytest.mark.parametrize(
    "gradient_with_incorrect_signature",
    [
        (lambda direction, x, y, z: direction),
        (lambda direction, x: direction),
        (lambda direction, y, x: direction),
        (lambda direction, x, w: direction),
    ],
)
def test_wrong_gradient_signature_raises_error_at_model_initialization(
    gradient_with_incorrect_signature,
):
    """Test that an error is raised if the gradient signature is wrong"""

    def forward(x, y):
        return x

    def gradient_with_correct_signature(direction, x, y):
        return direction

    cuqi.model.Model(
        forward, 1, (Discrete(1), Discrete(1)), gradient=gradient_with_correct_signature
    )

    with pytest.raises(
        ValueError,
        match=r"Gradient function signature should be \['direction', 'x', 'y'\]",
    ):
        cuqi.model.Model(
            forward,
            1,
            (Discrete(1), Discrete(1)),
            gradient=gradient_with_incorrect_signature,
        )

def test_model_raises_error_when_domain_geometry_is_inconsistent_with_forward_signature_at_initialization():
    """Test that the model raises an error if the domain geometry is inconsistent with the forward function signature at initialization"""

    def forward(a, b):
        return a * b

    with pytest.raises(
        ValueError,
        match=r"The forward operator input is specified by more than one argument. This is only supported for domain geometry of type tuple",
    ):
        cuqi.model.Model(forward, 1, 2)

def test_evaluating_model_at_distribution_with_non_unique_names_raises_error():
    """Test that an error is raised if the model is evaluated at distributions with non-unique parameter names"""

    def forward(a, b):
        return a*b

    model = cuqi.model.Model(forward, 1, (Discrete(1), Discrete(1)))

    # Create distributions with non-unique parameter names
    a = cuqi.distribution.Gaussian(0, 1, name="x")
    b = cuqi.distribution.Gaussian(0, 1, name="x")

    with pytest.raises(
        ValueError,
        match=r"Attempting to match parameter name of Model with given distributions, but distribution names are not unique. Please provide unique names for the distributions.",
    ):
        model(a, b)

def test_evaluating_model_at_random_variables_with_non_unique_names_raises_error():
    """Test that an error is raised if the model is evaluated at random variables with non-unique parameter names"""

    def forward(a, b):
        return a*b

    model = cuqi.model.Model(forward, 1, (Discrete(1), Discrete(1)))

    # Create distributions with non-unique parameter names
    a = cuqi.distribution.Gaussian(0, 1, name="x").rv
    b = cuqi.distribution.Gaussian(0, 1, name="x").rv

    with pytest.raises(
        ValueError,
        match=r"Attempting to match parameter name of Model with given random variables, but random variables names are not unique. Please provide unique names for the random variables.",
    ):
        model(a, b)

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
        """Test that the forward method can handle multiple inputs and return
        the correct output type (even if order of kwargs inputs is switched)"""

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
            model.forward(x, y=y, z=z)

class MultipleInputTestModel:
    """Class representing a test model with ingredients to set up variations of the model. For example, model with gradient, or with jacobian, etc. All the variations shares the same forward map, the domain and range geometry."""

    def __init__(self):
        self.model_class = None  # the cuqi.model class to set up the model
        self.forward_map = None
        self.pde = None  # in case of PDE model
        self.gradient_form1 = None  # callable
        self.gradient_form2 = None  # tuple of callables
        self.gradient_form2_incomplete = (
            None  # tuple of callables with some None elements
        )
        self.jacobian_form1 = None  # callable
        self.jacobian_form2 = None  # tuple of callables
        self.jacobian_form2_incomplete = (
            None  # tuple of callables with some None elements
        )
        self.domain_geometry = None
        self.range_geometry = None
        self.test_data = []  # list of TestCase objects holding the test data
        self.model_variations = (
            []
        )  # list of model variations which all share the same forward map
        self.input_bounds = None  # bounds for the test input values

    @property
    def has_gradient(self):
        """Check if the MultipleInputTestModel object has a way of determining 
        the model gradient. That is, if this object has gradient or jacobian
        (which can be used to specify the gradient) function of any form"""
        return (
            self.gradient_form1 is not None
            or self.gradient_form2 is not None
            or self.gradient_form2_incomplete is not None
            or self.jacobian_form1 is not None
            or self.jacobian_form2 is not None
            or self.jacobian_form2_incomplete is not None
        )

    def populate_model_variations(self):
        """Populate the `model_variations` list with different variations of the model that share the same forward map but differ in some other aspect like gradient, jacobian, etc."""

        if self.pde is not None:
            first_kwarg = {"PDE": self.pde}
        else:
            first_kwarg = {"forward": self.forward_map}

        # Model with forward only
        model = self.model_class(
            **first_kwarg,
            domain_geometry=self.domain_geometry,
            range_geometry=self.range_geometry
        )
        model._do_test_gradient = False  # do not test this model for gradient
        self.model_variations.append(model)

        # Model with gradient of from 1 (callable)
        if self.gradient_form1 is not None:
            model = self.model_class(
                **first_kwarg,
                gradient=self.gradient_form1,
                domain_geometry=self.domain_geometry,
                range_geometry=self.range_geometry,
            )
            self.model_variations.append(model)

        # Model with gradient of from 2 (tuple of callables)
        if self.gradient_form2 is not None:
            model = self.model_class(
                **first_kwarg,
                gradient=self.gradient_form2,
                domain_geometry=self.domain_geometry,
                range_geometry=self.range_geometry,
            )
            self.model_variations.append(model)

        # Model with gradient of from 2 incomplete (tuple of callables with some None elements)
        if self.gradient_form2_incomplete is not None:
            model = self.model_class(
                **first_kwarg,
                gradient=self.gradient_form2_incomplete,
                domain_geometry=self.domain_geometry,
                range_geometry=self.range_geometry,
            )
            self.model_variations.append(model)

        # Model with jacobian of from 1 (callable)
        if self.jacobian_form1 is not None:
            model = self.model_class(
                **first_kwarg,
                jacobian=self.jacobian_form1,
                domain_geometry=self.domain_geometry,
                range_geometry=self.range_geometry,
            )
            self.model_variations.append(model)

        # Model with jacobian of from 2 (tuple of callables)
        if self.jacobian_form2 is not None:
            model = self.model_class(
                **first_kwarg,
                jacobian=self.jacobian_form2,
                domain_geometry=self.domain_geometry,
                range_geometry=self.range_geometry,
            )
            self.model_variations.append(model)

        # Model with jacobian of from 2 incomplete (tuple of callables with some None elements)
        if self.jacobian_form2_incomplete is not None:
            model = self.model_class(
                **first_kwarg,
                jacobian=self.jacobian_form2_incomplete,
                domain_geometry=self.domain_geometry,
                range_geometry=self.range_geometry,
            )
            self.model_variations.append(model)

    @staticmethod
    def create_model_test_case_combinations():
        """Create all combinations of test model variations and test cases
        (test data) for the test models."""
        model_test_case_combinations = []
        test_model_list = []

        # Model 1
        test_model = MultipleInputTestModel.helper_build_three_input_test_model()
        test_model.populate_model_variations()
        TestCase.create_test_cases_for_test_model(test_model)
        test_model_list.append(test_model)

        # Model 2
        test_model = MultipleInputTestModel.helper_build_steady_state_PDE_test_model()
        test_model.populate_model_variations()
        TestCase.create_test_cases_for_test_model(test_model)
        test_model_list.append(test_model)

        # Model 3
        test_model = (
            MultipleInputTestModel.helper_build_two_input_test_with_mapped_parameter_model()
        )
        test_model.populate_model_variations()
        test_model.input_bounds = [
            0.1,
            0.9,
        ]  # choose input from uniform distribution in [0.1, 0.9]
        TestCase.create_test_cases_for_test_model(test_model)
        test_model_list.append(test_model)

        # Model 4
        test_model = MultipleInputTestModel.helper_build_time_dependent_PDE_test_model()
        test_model.populate_model_variations()
        test_model.input_bounds = [
            0.1,
            4,
        ]  # choose input from uniform distribution in [0.1, 4]
        TestCase.create_test_cases_for_test_model(test_model)
        test_model_list.append(test_model)

        # Append all combinations of test model variations and test cases
        # to model_test_case_combinations
        for test_model in test_model_list:
            for test_model_variation in test_model.model_variations:
                for test_data_item in test_model.test_data:
                    model_test_case_combinations.append(
                        (test_model_variation, test_data_item)
                    )

        return model_test_case_combinations

    @staticmethod
    def helper_build_three_input_test_model():
        """Build a MultipleInputTestModel with three inputs: x, y, z, and gradient and jacobian functions."""
        test_model = MultipleInputTestModel()
        test_model.forward_map = lambda x, y, z: x * y[0] + z * y[1]

        # gradient with respect to x
        def gradient_x(direction, x, y, z):
            return direction * y[0]

        # gradient with respect to y
        def gradient_y(direction, x, y, z):
            return np.array([direction @ x, direction @ z])

        # gradient with respect to z
        def gradient_z(direction, x, y, z):
            return direction * y[1]

        # gradient with respect to all inputs (form 1, callable)
        def gradient_form1(direction, x, y, z):
            grad_x = gradient_x(direction, x, y, z)
            grad_y = gradient_y(direction, x, y, z)
            grad_z = gradient_z(direction, x, y, z)
            return (grad_x, grad_y, grad_z)

        # Assign the gradient functions to the test model
        test_model.gradient_form1 = gradient_form1
        test_model.gradient_form2 = (gradient_x, gradient_y, gradient_z)
        test_model.gradient_form2_incomplete = (gradient_x, None, gradient_z)

        # jacobian with respect to x
        def jacobian_x(x, y, z):
            ones = np.ones_like(x)
            return np.diag(y[0] * ones)

        # jacobian with respect to y
        def jacobian_y(x, y, z):
            return np.array([x, z]).T

        # jacobian with respect to z
        def jacobian_z(x, y, z):
            ones = np.ones_like(x)
            return np.diag(y[1] * ones)

        # jacobian with respect to all inputs (form 1, callable)
        def jacobian_form1(x, y, z):
            jac_x = jacobian_x(x, y, z)
            jac_y = jacobian_y(x, y, z)
            jac_z = jacobian_z(x, y, z)
            return (jac_x, jac_y, jac_z)

        # Assign the jacobian functions to the test model
        test_model.jacobian_form1 = jacobian_form1
        test_model.jacobian_form2 = (jacobian_x, jacobian_y, jacobian_z)
        test_model.jacobian_form2_incomplete = (None, jacobian_y, jacobian_z)

        test_model.domain_geometry = (
            cuqi.geometry.Continuous1D(3),
            cuqi.geometry.Continuous1D(2),
            cuqi.geometry.Continuous1D(3),
        )
        test_model.range_geometry = cuqi.geometry.Continuous1D(3)

        test_model.model_class = cuqi.model.Model
        return test_model

    @staticmethod
    def helper_build_steady_state_PDE_test_model():
        """Build a PDE model with a steady state Poisson equation and two inputs: mag and kappa_scale."""

        # Poisson equation setup
        dim = 20  # Number of nodes
        L = 20  # Length of the domain
        dx = L / (dim - 1)  # grid spacing
        grid_sol = np.linspace(dx, L, dim - 1, endpoint=False)  # solution grid
        grid_obs = grid_sol[5:]  # observation grid
        source = lambda mag: mag * np.sin(
            grid_sol
        )  # source term depending on magnitude `mag`
        kappa = np.ones(dim)  # kappa is the diffusivity

        # Build the differential operator (depending on kappa_scale, a scale of the diffusivity)
        FOFD_operator = (
            cuqi.operator.FirstOrderFiniteDifference(dim - 1, bc_type="zero", dx=dx)
            .get_matrix()
            .todense()
        )
        diff_operator = (
            lambda kappa_scale: FOFD_operator.T
            @ np.diag(kappa_scale * kappa)
            @ FOFD_operator
        )

        # Build the PDE form
        poisson_form = lambda mag, kappa_scale: (
            diff_operator(kappa_scale),
            source(mag),
        )

        # Build the PDE object
        CUQI_pde = cuqi.pde.SteadyStateLinearPDE(
            poisson_form,
            grid_sol=grid_sol,
            grid_obs=grid_obs,
            observation_map=lambda u: u**2,
        )

        # Build the test model
        test_model = MultipleInputTestModel()
        test_model.model_class = cuqi.model.PDEModel
        test_model.pde = CUQI_pde
        test_model.domain_geometry = (Discrete(["mag"]), Discrete(["kappa_scale"]))
        test_model.range_geometry = Continuous1D(len(grid_obs))

        # Gradient with respect to mag
        def gradient_mag(direction, mag, kappa_scale):
            def fwd_mag(mag_):
                CUQI_pde.assemble(mag_, kappa_scale)
                u, _ = CUQI_pde.solve()
                obs_u = CUQI_pde.observe(u)
                return obs_u
            mag = mag.to_numpy() if isinstance(mag, CUQIarray) else mag 
            return direction @ cuqi.utilities.approx_derivative(fwd_mag, mag)

        # Gradient with respect to kappa_scale
        def gradient_kappa_scale(direction, mag, kappa_scale):
            def fwd_kappa_scale(kappa_scale_):
                CUQI_pde.assemble(mag, kappa_scale_)
                u, _ = CUQI_pde.solve()
                obs_u = CUQI_pde.observe(u)
                return obs_u
            kappa_scale = kappa_scale.to_numpy() if isinstance(kappa_scale, CUQIarray) else kappa_scale
            return direction @ cuqi.utilities.approx_derivative(fwd_kappa_scale, kappa_scale)

        # Gradient with respect to all inputs (form 1, callable)
        def gradient_form1(direction, mag, kappa_scale):
            grad_mag = gradient_mag(direction, mag, kappa_scale)
            grad_kappa_scale = gradient_kappa_scale(direction, mag, kappa_scale)
            return (grad_mag, grad_kappa_scale)

        # Assign the gradient functions to the test model
        test_model.gradient_form1 = gradient_form1
        test_model.gradient_form2 = (gradient_mag, gradient_kappa_scale)
        test_model.gradient_form2_incomplete = (gradient_mag, None)

        # Jacobian with respect to mag
        def jacobian_mag(mag, kappa_scale):
            def fwd_mag(mag_):
                CUQI_pde.assemble(mag_, kappa_scale)
                u, _ = CUQI_pde.solve()
                obs_u = CUQI_pde.observe(u)
                return obs_u
            mag = mag.to_numpy() if isinstance(mag, CUQIarray) else mag
            return cuqi.utilities.approx_derivative(fwd_mag, mag).reshape(-1, 1)

        # Jacobian with respect to kappa_scale
        def jacobian_kappa_scale(mag, kappa_scale):
            def fwd_kappa_scale(kappa_scale_):
                CUQI_pde.assemble(mag, kappa_scale_)
                u, _ = CUQI_pde.solve()
                obs_u = CUQI_pde.observe(u)
                return obs_u
            kappa_scale = kappa_scale.to_numpy() if isinstance(kappa_scale, CUQIarray) else kappa_scale
            return cuqi.utilities.approx_derivative(fwd_kappa_scale, kappa_scale).reshape(-1, 1)

        # Jacobian with respect to all inputs (form 1, callable)
        def jacobian_form1(mag, kappa_scale):
            jac_mag = jacobian_mag(mag, kappa_scale)
            jac_kappa_scale = jacobian_kappa_scale(mag, kappa_scale)
            return (jac_mag, jac_kappa_scale)

        # Assign the jacobian functions to the test model
        test_model.jacobian_form1 = jacobian_form1
        test_model.jacobian_form2 = (jacobian_mag, jacobian_kappa_scale)
        test_model.jacobian_form2_incomplete = (jacobian_mag, None)

        return test_model

    @staticmethod
    def helper_build_two_input_test_with_mapped_parameter_model():
        """Build a model with two inputs in which the inputs are mapped
        via geometry mapping"""
        test_model = MultipleInputTestModel()
        test_model.forward_map = lambda a, b: np.array(
            [a[0] ** 2 * b[0] * b[1], b[1] * a[0] * a[1]]
        )

        # gradient with respect to a
        def gradient_a(direction, a, b):
            return np.array(
                [
                    2 * a[0] * b[0] * b[1] * direction[0] + b[1] * a[1] * direction[1],
                    b[1] * a[0] * direction[1],
                ]
            )

        # gradient with respect to b
        def gradient_b(direction, a, b):
            return np.array(
                [
                    a[0] ** 2 * b[1] * direction[0],
                    a[0] ** 2 * b[0] * direction[0] + a[0] * a[1] * direction[1],
                ]
            )

        # gradient with respect to all inputs (form 1, callable)
        def gradient_form1(direction, a, b):
            grad_a = gradient_a(direction, a, b)
            grad_b = gradient_b(direction, a, b)
            return (grad_a, grad_b)

        test_model.gradient_form1 = gradient_form1
        test_model.gradient_form2 = (gradient_a, gradient_b)
        test_model.gradient_form2_incomplete = (gradient_a, None)

        test_model.jacobian_form1 = None
        test_model.jacobian_form2 = None
        test_model.jacobian_form2_incomplete = None

        # Setting up domain and range geometries
        geom_a = cuqi.geometry.MappedGeometry(
            cuqi.geometry.Continuous1D(2),
            map=lambda x: np.sin(x),
            imap=lambda x: np.arcsin(x),
        )
        geom_a.gradient = lambda direction, x: direction @ np.diag(np.cos(x))

        geom_b = cuqi.geometry.MappedGeometry(
            cuqi.geometry.Continuous1D(2),
            map=lambda x: x**3,
            imap=lambda x: x ** (1 / 3),
        )
        geom_b.gradient = lambda direction, x: 3 * direction @ np.diag(x**2)
        test_model.domain_geometry = (geom_a, geom_b)
        test_model.range_geometry = cuqi.geometry.Continuous1D(2)

        # Assign the model class
        test_model.model_class = cuqi.model.Model
        return test_model

    @staticmethod
    def helper_build_time_dependent_PDE_test_model():
        """Build a PDE model with a time-dependent PDE and two inputs: mag, and IC."""

        # Prepare PDE form
        N = 20   # Number of solution nodes
        endpoint = 1.0 # Length of the domain
        max_time = 0.1 # Maximum time
        dx = endpoint/(N+1)   # space step size
        cfl = 5/11 # the cfl condition to have a stable solution
        dt_approx = cfl*dx**2 # defining approximate time step size
        max_iter = int(max_time/dt_approx) # number of time steps
        Dxx_matr = (np.diag( -2*np.ones(N) ) + np.diag(np.ones(N-1),-1) + np.diag(np.ones(N-1),1))/dx**2
        Dxx = lambda mag: mag * Dxx_matr # FD diffusion operator

        # Grids for model
        grid_domain = np.linspace(dx, endpoint, N, endpoint=False)
        grid_range = np.linspace(dx, endpoint, N, endpoint=False) 
        time_steps = np.linspace(0,max_time,max_iter+1,endpoint=True)

        # PDE form (mag, IC, time)
        def PDE_form(mag, IC, t): return (Dxx(mag), np.zeros(N), IC)
        PDE = cuqi.pde.TimeDependentLinearPDE(
            PDE_form, time_steps, grid_sol=grid_domain, grid_obs=grid_range, method='backward_euler')

        # Build the test model
        test_model = MultipleInputTestModel()
        test_model.model_class = cuqi.model.PDEModel
        test_model.pde = PDE
        test_model.domain_geometry = (Discrete(["mag"]), Continuous1D(grid_domain))
        test_model.range_geometry = Continuous1D(grid_range)
        
        test_model.model_class = cuqi.model.PDEModel

        # Gradient with respect to mag
        def gradient_mag(direction, mag, IC):
            def fwd_mag(mag_):
                PDE.assemble(mag_, IC)
                u, _ = PDE.solve()
                obs_u = PDE.observe(u)
                return obs_u
            mag = mag.to_numpy() if isinstance(mag, CUQIarray) else mag 
            return direction @ cuqi.utilities.approx_derivative(fwd_mag, mag)
        
        # Gradient with respect to IC
        def gradient_IC(direction, mag, IC):
            def fwd_IC(IC_):
                PDE.assemble(mag, IC_)
                u, _ = PDE.solve()
                obs_u = PDE.observe(u)
                return obs_u
            IC = IC.to_numpy() if isinstance(IC, CUQIarray) else IC 
            return direction @ cuqi.utilities.approx_derivative(fwd_IC, IC)
        
        # Gradient with respect to all inputs (form 1, callable)
        def gradient_form1(direction, mag, IC):
            grad_mag = gradient_mag(direction, mag, IC)
            grad_IC = gradient_IC(direction, mag, IC)
            return (grad_mag, grad_IC)
        
        # Assign the gradient functions to the test model
        test_model.gradient_form1 = gradient_form1
        test_model.gradient_form2 = (gradient_mag, gradient_IC)
        test_model.gradient_form2_incomplete = (gradient_mag, None)

        # Jacobian with respect to mag
        def jacobian_mag(mag, IC):
            def fwd_mag(mag_):
                PDE.assemble(mag_, IC)
                u, _ = PDE.solve()
                obs_u = PDE.observe(u)
                return obs_u
            mag = mag.to_numpy() if isinstance(mag, CUQIarray) else mag 
            return cuqi.utilities.approx_derivative(fwd_mag, mag)
        
        # Jacobian with respect to IC
        def jacobian_IC(mag, IC):
            def fwd_IC(IC_):
                PDE.assemble(mag, IC_)
                u, _ = PDE.solve()
                obs_u = PDE.observe(u)
                return obs_u
            IC = IC.to_numpy() if isinstance(IC, CUQIarray) else IC 
            return cuqi.utilities.approx_derivative(fwd_IC, IC)
        
        # Jacobian with respect to all inputs (form 1, callable)
        def jacobian_form1(mag, IC):
            jac_mag = jacobian_mag(mag, IC)
            jac_IC = jacobian_IC(mag, IC)
            return (jac_mag, jac_IC)
        
        # Assign the jacobian functions to the test model
        test_model.jacobian_form1 = jacobian_form1
        test_model.jacobian_form2 = (jacobian_mag, jacobian_IC)
        test_model.jacobian_form2_incomplete = (jacobian_mag, None)

        return test_model

class TestCase:
    """Class representing a test case for a test model. A test case consists of the input values, the expected output values, and the expected output types and error messages."""

    def __init__(self, test_model):
        self._test_model = test_model

        # Non default arguments of the cuqi.model.Model object
        self._non_default_args = self._test_model.model_variations[0]._non_default_args
        self.forward_input = None
        self.forward_input_stacked = None
        self.direction = None
        self.expected_fwd_output = None
        self.expected_grad_output_value = None
        self.expected_staked_grad_output_value = None
        self.expected_fwd_output_type = None
        self.expected_grad_output_type = None
        self.FD_grad_output = None  # finite difference gradient for verification

    @property
    def model_forward_map(self):
        """Returns the underlying forward map of the test model (the one that maps funvals to funvals). Note that for cuqi.model.Model objects, the underlying forward map, stored in the `_forward_func` attribute, maps funvals to funvals, while the `forward` method that wraps the underlying `_forward_func` maps parameter values to parameter values."""

        return self._test_model.model_variations[0]._forward_func

    def create_input(self):
        """Create random input values for the test case, stored in two formats, a kwargs dictionary and a stacked numpy array."""
        input_dict = {}  # kwargs dictionary

        for i, arg in enumerate(self._non_default_args):
            dim = self._test_model.domain_geometry[i].par_dim
            # If case: input bounds are specified, sample from uniform distribution
            if self._test_model.input_bounds is not None:
                input_dict[arg] = np.random.uniform(
                    [self._test_model.input_bounds[0]] * dim,
                    [self._test_model.input_bounds[1]] * dim,
                )
            # Else case: sample from standard normal distribution
            else:
                input_dict[arg] = np.random.randn(dim)

        self.forward_input = input_dict
        self.forward_input_stacked = np.hstack([v for v in list(input_dict.values())])

    def create_direction(self):
        """Create a random direction for the gradient computation."""
        self.direction = np.random.randn(self._test_model.range_geometry.par_dim)

    def compute_expected_fwd_output(self):
        """Compute the expected output of the forward map for the test case. Note that this does not use the method `forward` of the `cuqi.model.Model` object, but the underlying forward map stored in the `_forward_func` attribute which is the function supplied by the user (in Model class case) or built from the PDE object (in PDEModel case)."""
        mapped_input = self._compute_mapped_input(
            self.forward_input, self._test_model.domain_geometry
        )
        self.expected_fwd_output = self.model_forward_map(**mapped_input)

    @staticmethod
    def _compute_mapped_input(forward_input, domain_geom):
        """Map the input values to function values using the par2fun method of the domain geometry."""
        mapped_input = {}
        for i, (k, v) in enumerate(forward_input.items()):
            if hasattr(domain_geom[i], "map"):
                mapped_input[k] = domain_geom[i].par2fun(v)
            elif type(domain_geom[i]) in _identity_geometries:
                mapped_input[k] = v
            else:
                raise NotImplementedError
        return mapped_input

    def compute_expected_grad_output(self):
        """Compute the expected output of the gradient computation for the test case. Note that this does not use the method `gradient` of the `cuqi.model.Model` object, but the underlying gradient functions supplied by the user."""

        domain_geom = self._test_model.domain_geometry

        # Map input values to function values
        forward_input_fun = deepcopy(self.forward_input)
        forward_input_fun = self._compute_mapped_input(self.forward_input, domain_geom)

        # Compute the expected gradient output
        self.expected_grad_output_value = []
        for i, (k, v) in enumerate(self.forward_input.items()):
            self.expected_grad_output_value.append(
                self._test_model.gradient_form2[i](self.direction, **forward_input_fun)
            )
            # If case: apply the chain rule for the gradient computation if the geometry has a gradient method
            if hasattr(domain_geom[i], "gradient"):
                self.expected_grad_output_value[-1] = domain_geom[i].gradient(
                    self.expected_grad_output_value[-1], v
                )
            elif type(domain_geom[i]) not in _identity_geometries:
                raise NotImplementedError
        self.expected_grad_output_value = tuple(self.expected_grad_output_value)
        self._compute_expected_stacked_grad_output()

    def _compute_expected_stacked_grad_output(self):
        """Compute the expected stacked gradient output for the test case. This is used for testing the gradient computation when the model flag _gradient_output_stacked is enabled."""
        expected_staked_grad_output = np.hstack(
            [
                (
                    v.to_numpy()
                    if isinstance(v, CUQIarray)
                    else force_ndarray(v, flatten=True)
                )
                for v in list(self.expected_grad_output_value)
            ]
        )
        self.expected_staked_grad_output_value = expected_staked_grad_output

    def compute_FD_grad_output(self):
        """Compute the finite difference gradient for the test case, for verification."""
        FD_grad_list = []
        model = self._test_model.model_variations[0]
        for k, v in self.forward_input.items():
            # Create forward lambda function with one input x, representing
            # the value of k to be used in computing the finite difference
            # gradient with respect to k
            forward_input = self.forward_input.copy()
            del forward_input[k]
            fwd = lambda x: model(**forward_input, **{k: x})

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

        if test_model.has_gradient:
            test_case.create_direction()
            test_case.compute_expected_grad_output()
            test_case.compute_FD_grad_output()
            test_case.expected_grad_output_type = np.ndarray

        test_model.test_data.append(test_case)

        # Case 2: all inputs are CUQIarrays
        test_case = TestCase(test_model)
        test_case.create_input()
        for i, (k, v) in enumerate(test_case.forward_input.items()):
            test_case.forward_input[k] = cuqi.array.CUQIarray(
                v, geometry=test_model.domain_geometry[i]
            )
        test_case.compute_expected_fwd_output()
        test_case.expected_fwd_output = cuqi.array.CUQIarray(
            test_case.expected_fwd_output, geometry=test_model.range_geometry
        )
        test_case.expected_fwd_output_type = cuqi.array.CUQIarray

        if test_model.has_gradient:
            test_case.create_direction()
            test_case.direction = cuqi.array.CUQIarray(
                test_case.direction, geometry=test_model.range_geometry
            )
            test_case.compute_expected_grad_output()
            test_case.compute_FD_grad_output()
            test_case.expected_grad_output_type = cuqi.array.CUQIarray

        test_model.test_data.append(test_case)

        # Case 3: inputs are mix of CUQIarrays and numpy arrays
        test_case = TestCase(test_model)
        test_case.create_input()
        for i, (k, v) in enumerate(test_case.forward_input.items()):
            if i == 0:
                test_case.forward_input[k] = cuqi.array.CUQIarray(
                    v, geometry=test_model.domain_geometry[i]
                )
        test_case.compute_expected_fwd_output()
        test_case.expected_fwd_output = cuqi.array.CUQIarray(
            test_case.expected_fwd_output, geometry=test_model.range_geometry
        )
        test_case.expected_fwd_output_type = cuqi.array.CUQIarray

        if test_model.has_gradient:
            test_case.create_direction()
            test_case.compute_expected_grad_output()
            test_case.compute_FD_grad_output()
            test_case.expected_grad_output_type = cuqi.array.CUQIarray

        test_model.test_data.append(test_case)

        # Case 4: same as previous case but direction is a CUQIarray
        test_case = deepcopy(test_case)
        if test_model.has_gradient:
            test_case.direction = cuqi.array.CUQIarray(
                test_case.direction, geometry=test_model.range_geometry
            )

        test_model.test_data.append(test_case)

        # Case 5: inputs are mix of CUQIarrays and samples (should raise an error)
        test_case = TestCase(test_model)
        test_case.create_input()
        for i, (k, v) in enumerate(test_case.forward_input.items()):
            if i == 0:
                v2 = 1.3 * v
                samples = np.vstack([v, v2]).T
                test_case.forward_input[k] = cuqi.samples.Samples(
                    samples, geometry=test_model.domain_geometry[i]
                )
            else:
                test_case.forward_input[k] = cuqi.array.CUQIarray(
                    v, geometry=test_model.domain_geometry[i]
                )
        test_case.expected_fwd_output = TypeError(
            "If applying the function to Samples, all inputs should be Samples."
        )

        if test_model.has_gradient:
            test_case.create_direction()
            test_case.expected_grad_output_value = NotImplementedError(
                "Gradient is not implemented for input of type Samples."
            )

        test_model.test_data.append(test_case)

        # Case 6: inputs are samples (should work for forward but not for gradient)
        test_case = TestCase(test_model)

        # Create model input as samples
        test_case.create_input()
        for i, (k, v) in enumerate(test_case.forward_input.items()):
            v2 = 1.5 * v
            samples = np.vstack([v, v2]).T
            test_case.forward_input[k] = cuqi.samples.Samples(
                samples, geometry=test_model.domain_geometry[i]
            )
        # Create stacked samples input (not supported for forward or gradient)
        # We use it in testing that it raises an error
        stacked_input_samples = []
        for i in range(2):
            stacked_input_samples.append(
                np.hstack([v.samples[:, i] for v in test_case.forward_input.values()])
            )
        test_case.forward_input_stacked = cuqi.samples.Samples(
            np.vstack([stacked_input_samples]).T, geometry=test_model.domain_geometry
        )

        # Compute expected forward output
        expected_fwd_output = []
        for i in range(2):
            input_i = {k: v.samples[:, i] for k, v in test_case.forward_input.items()}
            mapped_input_i = TestCase._compute_mapped_input(
                input_i, test_model.domain_geometry
            )
            expected_fwd_output.append(test_case.model_forward_map(**mapped_input_i))
        expected_fwd_output = np.vstack(expected_fwd_output).T
        test_case.expected_fwd_output = cuqi.samples.Samples(
            expected_fwd_output, geometry=test_model.range_geometry
        )
        test_case.expected_fwd_output_type = cuqi.samples.Samples

        if test_model.has_gradient:
            test_case.create_direction()
            test_case.expected_grad_output_value = NotImplementedError(
                "Gradient is not implemented for input of type Samples."
            )

        test_model.test_data.append(test_case)

        # Case 7: inputs are samples but of different length (should raise an error)
        test_case = TestCase(test_model)

        # Create model input as samples
        test_case.create_input()
        for i, (k, v) in enumerate(test_case.forward_input.items()):
            if i == 0:  # Case i=0: two samples
                v2 = 1.5 * v
                samples = np.vstack([v, v2]).T
                test_case.forward_input[k] = cuqi.samples.Samples(
                    samples, geometry=test_model.domain_geometry[i]
                )
            else:  # Case i>0: three samples
                v2 = 1.5 * v
                v3 = 2 * v
                samples = np.vstack([v, v2, v3]).T
                test_case.forward_input[k] = cuqi.samples.Samples(
                    samples, geometry=test_model.domain_geometry[i]
                )

        test_case.expected_fwd_output = ValueError(
            "If applying the function to Samples, all inputs should have the same number of samples."
        )

        if test_model.has_gradient:
            test_case.create_direction()
            test_case.expected_grad_output_value = NotImplementedError(
                "Gradient is not implemented for input of type Samples."
            )

        test_model.test_data.append(test_case)


# Create all combinations of test model variations and test cases and
# store them in model_test_case_combinations to be used in the tests
model_test_case_combinations = (
    MultipleInputTestModel.create_model_test_case_combinations()
)


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
        print("=" * 50)
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
        print("#" * 50)
        print("#" * 50)


@pytest.mark.parametrize("test_model, test_data", model_test_case_combinations)
def test_forward_of_multiple_input_model_is_correct(test_model, test_data):
    """Test that the forward method can handle multiple inputs when evaluated on the test data and return the correct output type and value"""
    assert isinstance(test_model, cuqi.model.Model)
    assert isinstance(test_data, TestCase)

    # If case: cases where we can apply the forward function without raising an error
    if not isinstance(
        test_data.expected_fwd_output, (NotImplementedError, TypeError, ValueError)
    ):
        fwd_output = test_model(**test_data.forward_input)

        # Assert output has the expected type
        assert isinstance(fwd_output, test_data.expected_fwd_output_type)

        # Assert output has the expected value
        if isinstance(fwd_output, np.ndarray):
            assert np.allclose(fwd_output, test_data.expected_fwd_output)
        elif isinstance(fwd_output, cuqi.samples.Samples):
            assert np.allclose(
                fwd_output.samples, test_data.expected_fwd_output.samples
            )
        else:
            raise NotImplementedError(
                "Checks for other types of outputs not implemented."
            )

    # Else case: cases where applying the forward function raises an error
    else:
        with pytest.raises(
            type(test_data.expected_fwd_output),
            match=str(test_data.expected_fwd_output),
        ):
            fwd_output = test_model(**test_data.forward_input)


# Create a sublist of model_test_case_combinations for cases where the expected
# forward output is not an error
model_test_case_combinations_no_forward_error = [
    (test_model, test_data)
    for test_model, test_data in model_test_case_combinations
    if not isinstance(
        test_data.expected_fwd_output, (NotImplementedError, TypeError, ValueError)
    )
]


@pytest.mark.parametrize("test_model, test_data", model_test_case_combinations_no_forward_error)
@pytest.mark.parametrize("funvals", [True, False])
def test_partial_forward_of_multiple_input_model_is_correct(test_model, test_data, funvals):
    """Test partial model evaluation for both parameter value and function value input"""
    assert isinstance(test_model, cuqi.model.Model)
    assert isinstance(test_data, TestCase)

    # Input kwargs
    kwargs =  test_data.forward_input
    # keys and values
    keys = list(kwargs.keys())
    values = list(kwargs.values())

    if funvals:
        # Convert all inputs to funvals
        values = [value.funvals if isinstance(value, CUQIarray)  else value for value in values]

    # Exclude case where expected output is Samples because it is not supported
    # in partial evaluation
    # Also exclude cases where the model gradient is not a tuple (will be tested
    # separately)
    if not isinstance(
        test_data.expected_fwd_output, (cuqi.samples.Samples)
    ) and isinstance(test_model._gradient_func, tuple):
        fwd_output_list = []

        # Evaluate all possible partial evaluations of the model
        if len(test_data.forward_input) == 2:
            fwd_output_list.append(test_model(**{keys[0]: values[0]})(**{keys[1]: values[1]}))
            fwd_output_list.append(test_model(values[0])(values[1])) # args instead of kwargs
            fwd_output_list.append(test_model(**{keys[1]: values[1]})(**{keys[0]: values[0]}))

        elif len(test_data.forward_input) == 3:
            fwd_output_list.append(test_model(**{keys[0]: values[0]})(**{keys[1]: values[1]})(**{keys[2]: values[2]}))
            fwd_output_list.append(test_model(values[0])(values[1])(values[2])) # args instead of kwargs
            fwd_output_list.append(test_model(**{keys[1]: values[1]})(**{keys[0]: values[0]})(**{keys[2]: values[2]}))
            fwd_output_list.append(test_model(**{keys[2]: values[2]})(**{keys[0]: values[0]})(**{keys[1]: values[1]}))
            fwd_output_list.append(test_model(**{keys[0]: values[0]})(**{keys[2]: values[2]})(**{keys[1]: values[1]}))
            fwd_output_list.append(test_model(**{keys[1]: values[1]})(**{keys[2]: values[2]})(**{keys[0]: values[0]}))
            fwd_output_list.append(test_model(**{keys[2]: values[2]})(**{keys[1]: values[1]})(**{keys[0]: values[0]}))

            fwd_output_list.append(test_model(**{keys[0]: values[0], keys[1]: values[1]})(**{keys[2]: values[2]}))
            fwd_output_list.append(test_model(values[0], values[1])(values[2])) # args instead of kwargs
            fwd_output_list.append(test_model(**{keys[1]: values[1], keys[0]: values[0]})(**{keys[2]: values[2]}))
            fwd_output_list.append(test_model(**{keys[2]: values[2], keys[0]: values[0]})(**{keys[1]: values[1]}))
            fwd_output_list.append(test_model(**{keys[0]: values[0], keys[2]: values[2]})(**{keys[1]: values[1]}))
            fwd_output_list.append(test_model(**{keys[1]: values[1], keys[2]: values[2]})(**{keys[0]: values[0]}))
            fwd_output_list.append(test_model(**{keys[2]: values[2], keys[1]: values[1]})(**{keys[0]: values[0]}))

        assert all(np.allclose(fwd_output_list[i], test_data.expected_fwd_output) for i in range(len(fwd_output_list)))

    # Case where the model gradient is not a tuple (attempting partial evaluation should raise NotImplementedError)
    elif not isinstance(test_model._gradient_func, tuple) and test_model._gradient_func is not None and not isinstance(test_data.expected_fwd_output, cuqi.samples.Samples):
        with pytest.raises(
            NotImplementedError,
            match=r"Partial forward model is only supported for gradient/jacobian functions that are tuples of callable functions"
        ):
            test_model(**{keys[0]: values[0]})


@pytest.mark.parametrize("test_model, test_data", model_test_case_combinations_no_forward_error)
def test_cannot_do_partial_forward_evaluation_with_samples_input(test_model, test_data):
    """Test attempting to partial evaluation of a model with Samples input
    creates a ValueError
    """
    assert isinstance(test_model, cuqi.model.Model)
    assert isinstance(test_data, TestCase)

    # Input kwargs
    kwargs =  test_data.forward_input
    # keys and values
    keys = list(kwargs.keys())
    values = list(kwargs.values())

    # Test only cases where expected output is Samples 
    # Also exclude cases where the model gradient is not a tuple (this case does
    # not support partial evaluation)
    if isinstance(
        test_data.expected_fwd_output, (cuqi.samples.Samples)
    ) and isinstance(test_model._gradient_func, tuple):

        if len(test_data.forward_input) == 2:
            with pytest.raises(
                ValueError,
                match=r"That is, partial evaluation or splitting is not supported for input of type Samples"
            ):
                test_model(**{keys[0]: values[0]})

            with pytest.raises(
                ValueError,
                match=r"That is, partial evaluation or splitting is not supported for input of type Samples"
            ): 
                test_model(values[0])
            pass

        elif len(test_data.forward_input) == 3:
            with pytest.raises(
                ValueError,
                match=r"That is, partial evaluation or splitting is not supported for input of type Samples"
            ):
                test_model(**{keys[0]: values[0]})

            with pytest.raises(
                ValueError,
                match=r"That is, partial evaluation or splitting is not supported for input of type Samples"
            ):
                test_model(**{keys[0]: values[0], keys[1]: values[1]})

            with pytest.raises(
                ValueError,
                match=r"That is, partial evaluation or splitting is not supported for input of type Samples"
            ):
                test_model(values[0], values[1])


@pytest.mark.parametrize(
    "test_model, test_data", model_test_case_combinations_no_forward_error
)
@pytest.mark.parametrize("funvals", [True, False])
def test_gradient_of_partial_forward_of_multiple_input_model_is_correct(
    test_model, test_data, funvals
):
    """Test gradient of partial model evaluation for both parameter value and function value input"""
    assert isinstance(test_model, cuqi.model.Model)
    assert isinstance(test_data, TestCase)

    # Input kwargs
    kwargs = test_data.forward_input
    # keys and values
    keys = list(kwargs.keys())
    values = list(kwargs.values())

    if funvals:
        # Convert all inputs to funvals
        values = [
            value.funvals if isinstance(value, CUQIarray) else value for value in values
        ]

    # Exclude case where expected output is Samples because it is not supported
    # in partial evaluation
    # Also exclude cases where the model gradient is not a tuple (will be tested
    # separately)
    if not isinstance(
        test_data.expected_fwd_output, (cuqi.samples.Samples)
    ) and isinstance(test_model._gradient_func, tuple):

        # Direction at which the gradient is evaluated
        direction = test_data.direction

        # Evaluate all possible partial evaluations of the model
        if len(test_data.forward_input) == 2:
            if test_model._gradient_func[1] is not None:
                grad = test_model(**{keys[0]: values[0]}).gradient(
                    direction, **{keys[1]: values[1]}
                )
                assert np.allclose(grad, test_data.expected_grad_output_value[1])
                # pass args (in expected order) instead of kwargs
                grad = test_model(values[0]).gradient(direction, values[1])
                assert np.allclose(grad, test_data.expected_grad_output_value[1])

            if test_model._gradient_func[0] is not None:
                grad = test_model(**{keys[1]: values[1]}).gradient(
                    direction, **{keys[0]: values[0]}
                )
                assert np.allclose(grad, test_data.expected_grad_output_value[0])

        elif len(test_data.forward_input) == 3:
            if test_model._gradient_func[2] is not None:
                grad = test_model(**{keys[0]: values[0]})(
                    **{keys[1]: values[1]}
                ).gradient(direction, **{keys[2]: values[2]})
                assert np.allclose(grad, test_data.expected_grad_output_value[2])
                # pass args (in expected order) instead of kwargs
                grad = test_model(values[0])(values[1]).gradient(direction, values[2])
                assert np.allclose(grad, test_data.expected_grad_output_value[2])

            if test_model._gradient_func[2] is not None:
                grad = test_model(**{keys[1]: values[1]})(
                    **{keys[0]: values[0]}
                ).gradient(direction, **{keys[2]: values[2]})
                assert np.allclose(grad, test_data.expected_grad_output_value[2])

            if test_model._gradient_func[1] is not None:
                grad = test_model(**{keys[2]: values[2]})(
                    **{keys[0]: values[0]}
                ).gradient(direction, **{keys[1]: values[1]})
                assert np.allclose(grad, test_data.expected_grad_output_value[1])

            if test_model._gradient_func[1] is not None:
                grad = test_model(**{keys[0]: values[0]})(
                    **{keys[2]: values[2]}
                ).gradient(direction, **{keys[1]: values[1]})
                assert np.allclose(grad, test_data.expected_grad_output_value[1])

            if test_model._gradient_func[0] is not None:
                grad = test_model(**{keys[1]: values[1]})(
                    **{keys[2]: values[2]}
                ).gradient(direction, **{keys[0]: values[0]})
                assert np.allclose(grad, test_data.expected_grad_output_value[0])

            if test_model._gradient_func[0] is not None:
                grad = test_model(**{keys[2]: values[2]})(
                    **{keys[1]: values[1]}
                ).gradient(direction, **{keys[0]: values[0]})
                assert np.allclose(grad, test_data.expected_grad_output_value[0])

            if test_model._gradient_func[2] is not None:
                grad = test_model(**{keys[0]: values[0], keys[1]: values[1]}).gradient(
                    direction, **{keys[2]: values[2]}
                )
                assert np.allclose(grad, test_data.expected_grad_output_value[2])
                # pass args (in expected order) instead of kwargs
                grad = test_model(values[0], values[1]).gradient(direction, values[2])
                assert np.allclose(grad, test_data.expected_grad_output_value[2])

            if test_model._gradient_func[2] is not None:
                grad = test_model(**{keys[1]: values[1], keys[0]: values[0]}).gradient(
                    direction, **{keys[2]: values[2]}
                )
                assert np.allclose(grad, test_data.expected_grad_output_value[2])

            if test_model._gradient_func[1] is not None:
                grad = test_model(**{keys[2]: values[2], keys[0]: values[0]}).gradient(
                    direction, **{keys[1]: values[1]}
                )
                assert np.allclose(grad, test_data.expected_grad_output_value[1])

            if test_model._gradient_func[1] is not None:
                grad = test_model(**{keys[0]: values[0], keys[2]: values[2]}).gradient(
                    direction, **{keys[1]: values[1]}
                )
                assert np.allclose(grad, test_data.expected_grad_output_value[1])

            if test_model._gradient_func[0] is not None:
                grad = test_model(**{keys[1]: values[1], keys[2]: values[2]}).gradient(
                    direction, **{keys[0]: values[0]}
                )
                assert np.allclose(grad, test_data.expected_grad_output_value[0])

            if test_model._gradient_func[0] is not None:
                grad = test_model(**{keys[2]: values[2], keys[1]: values[1]}).gradient(
                    direction, **{keys[0]: values[0]}
                )
                assert np.allclose(grad, test_data.expected_grad_output_value[0])

    # Case where the model gradient is not a tuple (attempting partial evaluation should raise NotImplementedError)
    elif (
        not isinstance(test_model._gradient_func, tuple)
        and test_model._gradient_func is not None
        and not isinstance(test_data.expected_fwd_output, cuqi.samples.Samples)
    ):
        with pytest.raises(
            NotImplementedError,
            match=r"Partial forward model is only supported for gradient/jacobian functions that are tuples of callable functions",
        ):
            test_model(**{keys[0]: values[0]})


@pytest.mark.parametrize(
    "test_model, test_data", model_test_case_combinations_no_forward_error
)
def test_forward_of_multiple_input_model_is_correct_when_input_is_stacked(
    test_model, test_data
):
    """Test that the forward method can handle stacked inputs and return the correct
    output type and value"""
    assert isinstance(test_model, cuqi.model.Model)
    assert isinstance(test_data, TestCase)

    # If case: the output is expected to be a numpy array
    if isinstance(test_data.expected_fwd_output, np.ndarray):
        assert len(test_data.forward_input_stacked) == test_model.domain_dim
        fwd_output_stacked_inputs = test_model.forward(test_data.forward_input_stacked)
        assert isinstance(fwd_output_stacked_inputs, np.ndarray)
        assert np.allclose(fwd_output_stacked_inputs, test_data.expected_fwd_output)
    # Else case: case where the input (and hence expected output) are Samples object.
    # For this case, the forward method should raise an error since applying the
    # forward function to a Samples object of stacked input is not supported.
    elif isinstance(test_data.expected_fwd_output, Samples):
        with pytest.raises(
            ValueError,
            match=r"When using Samples objects as input, the user should provide a Samples object for each non_default_args"
        ):
            test_model.forward(test_data.forward_input_stacked)
    else:
        raise NotImplementedError

    # Assert evaluating forward on stacked input with wrong dimension is treated
    # as a partial evaluation of the forward model
    if isinstance(test_data.expected_fwd_output, np.ndarray):
        if (
            not isinstance(test_model._gradient_func, tuple)
            and test_model._gradient_func is not None
        ):
            # Splitting is not possible here and this is interpreted as a
            # partial evaluation of the forward model which is not supported
            # when the gradient function is not a tuple of callable functions
            with pytest.raises(
                NotImplementedError,
                match=r"Partial forward model is only supported for gradient/jacobian functions that are tuples of callable functions"
            ):
                test_model.forward(test_data.forward_input_stacked[:-1])
        else: # No error expected (splitting is not performed but this is
            # interpreted as a partial evaluation of the forward model)
            test_model.forward(test_data.forward_input_stacked[:-1])


@pytest.mark.parametrize(
    "test_model, test_data", model_test_case_combinations_no_forward_error
)
def test_forward_of_multiple_input_model_applied_to_funvals_input_is_correct(
    test_model, test_data
):
    """Test that the forward method can handle multiple inputs some of which are
    function values and return the correct output"""

    # par input
    par_input = test_data.forward_input
    model_output_par_input = test_model(**par_input)

    # fun input (note because some of the test data cases have a mix of numpy
    # arrays and cuqi arrays, only the cuqi arrays are converted to funvals
    # below, hence it will be a case of mixed input of funvals and par values)
    fun_input = {
        k: v.funvals if (isinstance(v, CUQIarray) or isinstance(v, Samples)) else v
        for k, v in par_input.items()
    }
    model_output_fun_input = test_model(**fun_input)

    # Check that the output is the same
    if isinstance(model_output_par_input, np.ndarray):
        assert np.allclose(model_output_par_input, model_output_fun_input)
    elif isinstance(model_output_par_input, cuqi.samples.Samples):
        assert np.allclose(
            model_output_par_input.samples, model_output_fun_input.samples
        )


@pytest.mark.parametrize("test_model, test_data", model_test_case_combinations)
def test_gradient_of_multiple_input_model_is_correct(test_model, test_data):
    """Test that the gradient method can handle multiple inputs and
    return the correct output type and value"""

    assert isinstance(test_model, cuqi.model.Model)
    assert isinstance(test_data, TestCase)

    # If case: gradient is not implemented for the model
    if hasattr(test_model, "_do_test_gradient") and not test_model._do_test_gradient:
        with pytest.raises(
            NotImplementedError, match="Gradient is not implemented for this model."
        ):
            grad_output = test_model.gradient(
                test_data.direction, **test_data.forward_input
            )

    # Elif case: gradient is implemented for the model and expected to work
    elif not isinstance(
        test_data.expected_grad_output_value,
        (NotImplementedError, TypeError, ValueError),
    ):
        grad_output = test_model.gradient(
            test_data.direction, **test_data.forward_input
        )

        # Assert output format is a dictionary with expected keys
        assert list(grad_output.keys()) == test_model._non_default_args

        # Check type and value of the output
        for i, (k, v) in enumerate(grad_output.items()):
            # Verify that the output is of the expected type
            if v is not None:
                assert isinstance(v, test_data.expected_grad_output_type)

            # Verify that the output is of the expected value
            if (
                isinstance(test_model._gradient_func, tuple)
                and test_model._gradient_func[i] is None
            ):
                assert v is None
            else:
                assert np.allclose(v, test_data.expected_grad_output_value[i])
                assert np.allclose(v, test_data.FD_grad_output[i])

    # Else case: gradient is implemented for the model but expected to raise an error
    else:
        with pytest.raises(
            type(test_data.expected_grad_output_value),
            match=str(test_data.expected_grad_output_value),
        ):
            grad_output = test_model.gradient(
                test_data.direction, **test_data.forward_input
            )


# Create a sublist of model_test_case_combinations for cases where the expected
# gradient output is not an error
model_test_case_combinations_no_gradient_error = [
    (test_model, test_data)
    for test_model, test_data in model_test_case_combinations
    if not isinstance(
        test_data.expected_grad_output_value,
        (NotImplementedError, TypeError, ValueError),
    )
]
# Also, remove cases where the gradient is not implemented
model_test_case_combinations_no_gradient_error = [
    (test_model, test_data)
    for test_model, test_data in model_test_case_combinations_no_gradient_error
    if not hasattr(test_model, "_do_test_gradient") or test_model._do_test_gradient
]


@pytest.mark.parametrize(
    "test_model, test_data", model_test_case_combinations_no_gradient_error
)
def test_gradient_of_multiple_input_model_accepts_stacked_input(test_model, test_data):
    """Test that the gradient method can handle multiple inputs that are stacked and
    return the correct output type and value"""

    assert isinstance(test_model, cuqi.model.Model)
    assert isinstance(test_data, TestCase)

    grad_output_stacked_inputs = test_model.gradient(
        test_data.direction, test_data.forward_input_stacked
    )

    # Assert output format is a dictionary with expected keys
    assert list(grad_output_stacked_inputs.keys()) == test_model._non_default_args

    # Assert type and value of the output are correct
    for i, (k, v) in enumerate(grad_output_stacked_inputs.items()):
        if v is not None:
            assert isinstance(grad_output_stacked_inputs[k], np.ndarray)
            assert np.allclose(
                grad_output_stacked_inputs[k], test_data.expected_grad_output_value[i]
            )

    # Assert evaluating gradient on stacked input with wrong dimension raises 
    # error (it will be treated as a partial evaluation attempt of the gradient
    # method which is not supported)
    with pytest.raises(
        TypeError,
        match=r"missing (1|2) required positional argument(|s)",
    ):
        test_model.gradient(test_data.direction, test_data.forward_input_stacked[:-1])


@pytest.mark.parametrize(
    "test_model, test_data", model_test_case_combinations_no_gradient_error
)
def test_gradient_of_multiple_input_model_can_generate_stacked_output(
    test_model, test_data
):
    """Test that the gradient method can generate stacked output of correct
    value and type if the flag _gradient_output_stacked is enabled."""

    # Generate stacked output for cases where the gradient specification is
    # complete: all gradient functions are specified
    if callable(test_model._gradient_func) or None not in test_model._gradient_func:
        # Set the model flag _gradient_output_stacked to True to generate stacked output
        test_model._gradient_output_stacked = True
        stacked_grad_output = test_model.gradient(
            test_data.direction, **test_data.forward_input
        )
        # Reset the flag to False
        test_model._gradient_output_stacked = False

        # Assert type and value of the output are correct
        assert isinstance(stacked_grad_output, np.ndarray)
        assert np.allclose(
            stacked_grad_output, test_data.expected_staked_grad_output_value
        )


@pytest.mark.parametrize(
    "test_model, test_data", model_test_case_combinations_no_gradient_error
)
def test_gradient_of_multiple_input_model_is_correct_with_funvals_input(
    test_model, test_data
):
    """Test that the gradient method can handle multiple inputs some of which are
    function values and return the correct output"""

    # Par input
    par_input = test_data.forward_input
    grad_output_par_input = test_model.gradient(test_data.direction, **par_input)

    # Fun input
    fun_input = {
        k: v.funvals if (isinstance(v, CUQIarray) or isinstance(v, Samples)) else v
        for k, v in par_input.items()
    }
    grad_output_fun_input = test_model.gradient(test_data.direction, **fun_input)

    # Check that the output is the same
    for k, v in grad_output_par_input.items():
        if v is not None:
            assert np.allclose(v, grad_output_fun_input[k])
        else:
            assert grad_output_fun_input[k] is None


@pytest.mark.parametrize("seed",[(0),(1),(2)])
def test_LinearModel_getMatrix(seed):
    np.random.seed(seed)
    A = np.random.randn(10,7) #Random matrix

    model1 = cuqi.model.LinearModel(A)
    model2 = cuqi.model.LinearModel(lambda x : A@x, lambda y: A.T@y, range_geometry=A.shape[0], domain_geometry=A.shape[1])
    
    mat1 = model1.get_matrix() #Normal matrix
    mat2 = model2.get_matrix() #Sparse matrix (generated from functions)

    assert np.allclose(mat1,mat2.toarray())

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
    """Test that linear model automatically infers parameter names from the forward and adjoint functions and evaluation matches."""

    forward_x = lambda x: x
    forward_y = lambda y: y

    def forward_z(z, not_used=None):
        return z

    adjoint_w = lambda w: w
    adjoint_v = lambda v: v

    model_x_w = cuqi.model.LinearModel(forward_x, adjoint_w, 1, 1)
    model_y_v = cuqi.model.LinearModel(forward_y, adjoint_v, 1, 1)
    model_z_v = cuqi.model.LinearModel(forward_z, adjoint_v, 1, 1)

    A = np.array([[1]])
    model_mat = cuqi.model.LinearModel(A)  # Default parameter name is 'x'

    # First: forward checks
    # Check that the model has switched its parameter name
    assert model_x_w._non_default_args == ["x"]
    assert model_y_v._non_default_args == ["y"]
    assert model_z_v._non_default_args == ["z"]
    assert model_mat._non_default_args == ["x"]

    # Check that we can provide parameter names when evaluating the model
    assert model_x_w(x=1) == 1
    assert model_y_v(y=1) == 1
    assert model_z_v(z=1) == 1
    assert model_mat(x=np.ones(1)) == np.ones(1)  # With matrix it has to be numpy array

    # Check providing the wrong parameter name raises an error
    with pytest.raises(
        ValueError,
        match=r"The model input is specified by keywords arguments \['y'\] that does not match the non_default_args of the model \['x'\].",
    ):
        model_x_w(y=1)

    # And check that we can provide positional arguments
    assert model_x_w(1) == 1
    assert model_y_v(1) == 1
    assert model_z_v(1) == 1
    assert model_mat(np.ones(1)) == np.ones(1)

    # Check that matrix multiplication works
    assert model_x_w @ 1 == 1
    assert model_y_v @ 1 == 1
    assert model_z_v @ 1 == 1
    assert model_mat @ np.ones(1) == np.ones(1)

    # Second: adjoint checks
    # Check we can provide parameter names when evaluating the adjoint
    assert model_x_w.adjoint(w=1) == 1
    assert model_y_v.adjoint(v=1) == 1
    assert model_z_v.adjoint(v=1) == 1

    # Check providing the wrong parameter name raises an error
    with pytest.raises(
        ValueError,
        match=r"The adjoint input is specified by keywords arguments \['v'\] that does not match the non_default_args of the adjoint \['w'\].",
    ):
        model_x_w.adjoint(v=1)

    # And check that we can provide positional arguments
    assert model_x_w.adjoint(1) == 1
    assert model_y_v.adjoint(1) == 1
    assert model_z_v.adjoint(1) == 1

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
    """Test that AffineModel only accepts LinearModel, besides accepting
    a matrix or callable"""

    # Random vectors
    x = np.random.randn(model.domain_dim)
    b = np.random.randn(model.range_dim)

    with pytest.raises(
        TypeError,
        match=r"The linear operator should be a LinearModel object, a callable function or a matrix.",
    ):
        A_affine = cuqi.model.AffineModel(
            model,
            b,
            range_geometry=model.range_geometry,
            domain_geometry=model.domain_geometry,
        )


@pytest.mark.parametrize("fwd_func", [lambda: 1, lambda x, y: x])
def test_AffineModel_raise_error_when_forward_or_adjoint_has_too_few_or_too_many_inputs(
    fwd_func,
):
    """Test that AffineModel raises an error when the forward function or the adjoint function has too few or too many inputs"""
    adj_func = lambda x: x

    # Forward function with too few/many inputs
    with pytest.raises(
        ValueError, match=r"The linear operator should have exactly one input argument"
    ):
        cuqi.model.AffineModel(fwd_func, 0, adj_func)

    # Adjoint function with too few/many inputs
    with pytest.raises(
        ValueError,
        match=r"The adjoint linear operator should have exactly one input argument",
    ):
        cuqi.model.AffineModel(adj_func, 0, fwd_func)

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


@pytest.mark.parametrize("domain_geometry, num_inputs, expected_domain_geometry", [
    (4, 1, _DefaultGeometry1D(4)),
    ((4,), 1, _DefaultGeometry1D(4)),
    (Discrete(4), 1, Discrete(4)),
    ((Discrete(4),), 1, Discrete(4)),
    ((2, 2), 1, _DefaultGeometry2D((2, 2))),
    (((2, 2),), 1, _DefaultGeometry2D((2, 2))),
    (Image2D((2,2)), 1, Image2D((2,2))),
    ((Image2D((2,2)),), 1, Image2D((2,2))),
    ((Discrete(5), 3), 2, _ProductGeometry(Discrete(5), _DefaultGeometry1D(3))),
    ((5, 3), 2, _ProductGeometry(_DefaultGeometry1D(5), _DefaultGeometry1D(3))),
    ((5, Discrete(3),), 2, _ProductGeometry(_DefaultGeometry1D(5), Discrete(3)))
])
@pytest.mark.parametrize("range_geometry, expected_range_geometry", [
    (1, _DefaultGeometry1D(1)),
    ((1,), _DefaultGeometry1D(1)),
    (Discrete(1), Discrete(1)),
    ((Discrete(1),), Discrete(1))
])
def test_setting_domain_and_range_geometry(domain_geometry, range_geometry, num_inputs, expected_domain_geometry, expected_range_geometry):
    """Test that the model can be initialized with different types of domain and range geometries"""
    # Set up the forward map for 1 input
    forward = lambda x: np.sum(x)
    # Set up the forward map for 2 inputs
    forward_2_inputs = lambda x, y: np.sum(x) + np.sum(y)

    # Create the model with the specified domain and range geometries
    if num_inputs == 1:
        model = cuqi.model.Model(
            forward,
            domain_geometry=domain_geometry,
            range_geometry=range_geometry,
        )
    else:
        model = cuqi.model.Model(
            forward_2_inputs,
            domain_geometry=domain_geometry,
            range_geometry=range_geometry,
        )

    # Check the model domain and range geometries are of the expected type
    assert model.domain_geometry == expected_domain_geometry
    assert model.range_geometry == expected_range_geometry

def test_partial_model_evaluation_with_distributions_not_supported():
    """ Test that partial evaluation of a model with multiple inputs is not
    supported when inputs are distributions."""
    # Create a model with multiple inputs
    test_model = MultipleInputTestModel.helper_build_three_input_test_model()
    model = cuqi.model.Model(
        test_model.forward_map,
        gradient=test_model.gradient_form2,
        domain_geometry=test_model.domain_geometry,
        range_geometry=test_model.range_geometry,
    )

    # Input values for the model
    x_dist = cuqi.distribution.Gaussian(np.array([1, 2, 3]), 1)
    y_dist = cuqi.distribution.Gaussian(np.array([4, 5]), 1)
    z_dist = cuqi.distribution.Gaussian(np.array([6, 7, 8]), 1)

    # This should work
    model_dist = model(x_dist, y_dist, z_dist)

    # This should not work
    with pytest.raises(ValueError,
                       match="Partial evaluation of the model is not supported for distributions"):
        model(x_dist)

def test_partial_model_evaluation_with_random_variables_not_supported():
    """ Test that partial evaluation of a model with multiple inputs is not
    supported when inputs are random variables."""
    # Create a model with multiple inputs
    test_model = MultipleInputTestModel.helper_build_three_input_test_model()
    model = cuqi.model.Model(
        test_model.forward_map,
        gradient=test_model.gradient_form2,
        domain_geometry=test_model.domain_geometry,
        range_geometry=test_model.range_geometry,
    )

    # Input values for the model
    x_rv = cuqi.distribution.Gaussian(np.array([1, 2, 3]), 1).rv
    y_rv = cuqi.distribution.Gaussian(np.array([4, 5]), 1).rv
    z_rv = cuqi.distribution.Gaussian(np.array([6, 7, 8]), 1).rv

    # This should work
    model_rv = model(x_rv, y_rv, z_rv)

    # This should not work
    with pytest.raises(ValueError,
                       match="Partial evaluation of the model is not supported for random variables"):
        model(x_rv)

def test_reduction_of_number_of_model_inputs_by_partial_specification():
    # Create a model with multiple inputs
    test_model = MultipleInputTestModel.helper_build_three_input_test_model()
    model = cuqi.model.Model(
        test_model.forward_map,
        gradient=test_model.gradient_form2,
        domain_geometry=test_model.domain_geometry,
        range_geometry=test_model.range_geometry,
    )

    # Input values for the model
    x_val = np.array([1, 2, 3])
    y_val = np.array([4, 5])
    z_val = np.array([6, 7, 8])

    # Check partial specification of model inputs generates a model with
    # reduced inputs
    model_given_x = model(x=x_val)
    assert list(model_given_x._non_default_args)==['y', 'z']
    model_given_y = model(y=y_val)
    assert list(model_given_y._non_default_args)==['x', 'z']
    model_given_z = model(z=z_val)
    assert list(model_given_z._non_default_args)==['x', 'y']
    model_given_xy = model(x=x_val, y=y_val)
    assert list(model_given_xy._non_default_args)==['z']
    model_given_yz = model(y=y_val, z=z_val)
    assert list(model_given_yz._non_default_args)==['x']
    model_given_xz = model(x=x_val, z=z_val)
    assert list(model_given_xz._non_default_args)==['y']

    # Check all ways of specifying inputs generate the same output
    output = model(x=x_val, y=y_val, z=z_val)
    assert(np.allclose(model(x=x_val)(y=y_val, z=z_val), output))
    assert(np.allclose(model(x=x_val)(z=z_val, y=y_val), output))
    assert(np.allclose(model(y=y_val)(x=x_val, z=z_val), output))
    assert(np.allclose(model(y=y_val)(z=z_val, x=x_val), output))
    assert(np.allclose(model(z=z_val)(x=x_val, y=y_val), output))
    assert(np.allclose(model(z=z_val)(y=y_val, x=x_val), output))
    assert(np.allclose(model(x=x_val, y=y_val)(z=z_val), output))
    assert(np.allclose(model(y=y_val, x=x_val)(z=z_val), output))
    assert(np.allclose(model(x=x_val, z=z_val)(y=y_val), output))
    assert(np.allclose(model(z=z_val, x=x_val)(y=y_val), output))
    assert(np.allclose(model(y=y_val, z=z_val)(x=x_val), output))
    assert(np.allclose(model(z=z_val, y=y_val)(x=x_val), output))
    assert(np.allclose(model(x=x_val)(y=y_val)(z=z_val), output))
