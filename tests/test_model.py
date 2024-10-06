from cuqi.array import CUQIarray
from cuqi.geometry import Continuous1D
import numpy as np
import scipy as sp
import cuqi
import pytest
from scipy import optimize
from copy import copy


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
                            wrt: A.T@direction,
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
                            wrt: A.T@direction,
                            domain_geometry=cuqi.geometry.Continuous1D(3),
                            range_geometry=cuqi.geometry.Continuous1D(2))

    if isinstance(direction, cuqi.samples.Samples):
        with pytest.raises(ValueError):
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
                            wrt: 2*wrt@A@direction,
                            domain_geometry=domain_geometry,
                            range_geometry=range_geometry)

    if case_id ==1:
        # Raises and error because imap is not implemented for
        # the domain_geometry and hence can't compute fun2par 
        # (and wrt is passed as function).
        with pytest.raises(ValueError):
            grad = model.gradient(direction, wrt, is_wrt_par=is_wrt_par)

    if case_id ==2:
        # Raises an error because gradient is not implemented for
        # the domain_geometry and the domain geometry is not in
        # the list cuqi.geometry._get_identity_geometries()
        with pytest.raises(NotImplementedError):
            grad = model.gradient(direction, wrt, is_wrt_par=is_wrt_par)

    if case_id ==3:
        # Raises an error because domain_geometry does not have an
        # implementation of fun2par and wrt is passed as function.
        with pytest.raises(NotImplementedError):
            grad = model.gradient(direction, wrt, is_wrt_par=is_wrt_par)

    if case_id == 4:
        # Raises an error because the range_geometry is not in the
        # cuqi.geometry._get_identity_geometries() list
        with pytest.raises(NotImplementedError):
            grad = model.gradient(direction, wrt, is_wrt_par=is_wrt_par)


@pytest.mark.parametrize("forward, gradient, direction, wrt, domain_geometry, domain_gradient, range_geometry, is_direction_par, is_wrt_par",
                         [
                             (
                                 lambda x: np.array([[1, 0, 0], [0, 3, .1]])@x,
                                 lambda direction, wrt: np.array(
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
                                 lambda direction, wrt: np.array(
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
def test_gradient_computation(forward, gradient, direction, wrt, domain_geometry, domain_gradient, range_geometry, is_direction_par, is_wrt_par):
    """Test applying chain rule when computing the gradient"""
    model = cuqi.model.Model(forward=forward,
                             gradient=gradient,
                             domain_geometry=domain_geometry,
                             range_geometry=range_geometry)
    model.domain_geometry.gradient = domain_gradient

    # Set CUQIarray geometries
    if isinstance(wrt, CUQIarray):
        wrt.geometry = domain_geometry
    if isinstance(direction, CUQIarray):
        direction.geometry = range_geometry
    # Compute cuqi model gradient 
    grad = model.gradient(direction, wrt, is_wrt_par=is_wrt_par)

    # If wrt is function value, convert wrt to parameters to
    # be passed as an input for to cuqi.utilities.approx_derivative
    if isinstance(wrt, CUQIarray):
        wrt_ndarray = np.zeros(len(wrt.parameters))
        wrt_ndarray[:] = wrt.parameters
        wrt = wrt_ndarray
    if not is_wrt_par:
        wrt = domain_geometry.fun2par(wrt)

    if isinstance(direction, CUQIarray):
        direction_ndarray = np.zeros(len(direction.parameters))
        direction_ndarray[:] = direction.parameters
        direction = direction_ndarray
    if not is_direction_par:
        direction = range_geometry.fun2par(direction)
    # Compute a finite difference approximation of cuqi model gradient 
    findiff_grad = cuqi.utilities.approx_derivative(
        model.forward, wrt, direction)

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

    with pytest.raises(ValueError, match=r"dimension does not match"):
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
    def gradient(dir, wrt):
        return dir@jacobian(wrt)
    
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
@pytest.mark.parametrize("model", [cuqi.testproblem.Deconvolution1D().model,
                                   cuqi.testproblem.Heat1D().model])
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


