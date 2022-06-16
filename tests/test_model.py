from cuqi.geometry import Continuous1D
import numpy as np
import scipy as sp
import cuqi
import pytest
from scipy import optimize


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
    assert( (model1.range_dim, model1.domain_dim) == (5,5) and model1.domain_geometry.shape == (5,) and
            len(model1.range_geometry.grid) == 5)

def test_model_allow_DefaultGeometry():
    """ Tests that model can have specific geometry (Image2D) and x can be _Defaultgeometry"""
    model = cuqi.testproblem.Deconvolution2D(dim=5).model
    x = cuqi.distribution.GaussianCov(np.zeros(model.domain_dim), 1).sample()
    model(x)   #Forward
    model.T(x) #Adjoint

@pytest.mark.parametrize("x, expected_type",
                         [(np.array([1, 3, 4]),
                           np.ndarray),

                          (cuqi.samples.Samples(
                              samples=np.array([[1, 3, 4],
                                               [5, 2, 6]]).T),
                           cuqi.samples.Samples),

                          (cuqi.samples.CUQIarray(np.array([1, 3, 4]),
                            geometry=cuqi.geometry.Continuous1D(3)),
                           cuqi.samples.CUQIarray)])
def test_forward(x, expected_type):
    """For different types of input to the model forward method, assert we are optaining the correct output type"""
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

                          (cuqi.samples.CUQIarray(np.array([1, 3]),
                            geometry=cuqi.geometry.Continuous1D(2)),
                           cuqi.samples.CUQIarray)])
def test_adjoint(x, expected_type):
    """For different types of input to the model adjoint method, assert we are optaining the correct output type"""
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

                          (cuqi.samples.CUQIarray(np.array([3, 4]),
                            geometry=cuqi.geometry.Continuous1D(2)),
                           cuqi.samples.CUQIarray)])
def test_gradient(direction, expected_type):
    """For different types of input to the model gradient method, assert we are optaining the correct output type"""
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

@pytest.mark.parametrize("forward, gradient, direction, wrt, domain_geometry, domain_gradient, range_geometry",
    [
        (
            lambda x: np.array([[1, 0, 0],[0, 3, .1]])@x,
            lambda direction, wrt: np.array([[1, 0, 0],[0, 3, .1]]).T@direction,
            np.array([1, 12]),
            np.array([1, 12, 8]),
            cuqi.geometry.MappedGeometry(Continuous1D(3), map=lambda x:2*x, imap=lambda x:x/2),
            lambda direction, wrt:2*np.eye(3)@direction,
            cuqi.geometry.Continuous1D(2)
        ),
        (
            lambda x: np.array([[1, 2, 8],[1, 3, .1]])@x,
            lambda direction, wrt: np.array([[1, 2, 8],[1, 3, .1]]).T@direction,
            np.array([1, 12]),
            np.array([1, 12, 8]),
            cuqi.geometry.MappedGeometry(Continuous1D(3), map=lambda x:x**2, imap=lambda x:np.sqrt(x)),
            lambda direction, wrt:2*np.diag(wrt)@direction,
            cuqi.geometry.Continuous1D(2)
        ),
        (
            lambda x: np.sin(x),
            lambda direction, wrt: np.diag(np.cos(wrt))@direction,
            np.array([1, 1, 4]),
            np.array([1, 12, 8]),
            cuqi.geometry.MappedGeometry(Continuous1D(3), map=lambda x:x**2, imap=lambda x:np.sqrt(x)),
            lambda direction, wrt:2*np.diag(wrt)@direction,
            cuqi.geometry.Continuous1D(3)
        ),
    ]
)
def test_gradient_computation(forward, gradient, direction, wrt, domain_geometry, domain_gradient, range_geometry):
    """Test applying chain rule when computing the gradient"""
    model = cuqi.model.Model(forward=forward,
                            gradient=gradient,
                            domain_geometry=domain_geometry,
                            range_geometry=range_geometry)
    model.domain_geometry.gradient=domain_gradient

      
    grad = model.gradient(direction, wrt)

    findiff_grad = cuqi.utilities.approx_derivative(model.forward, wrt, direction)
    assert(np.allclose(grad, findiff_grad))