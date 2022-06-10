import numpy as np
import scipy as sp
import cuqi
import pytest


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
    A = np.array(([1, 0, 0],[0, 3, .1]))

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
    A = np.array(([1, 0, 0],[0, 3, .1]))

    model = cuqi.model.LinearModel(forward=A,
                                   domain_geometry=cuqi.geometry.Continuous1D(3),
                                   range_geometry=cuqi.geometry.Continuous1D(2))

    ad = model.adjoint(x)
    assert(isinstance(ad, expected_type))
    pass

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
    A = np.array(([1, 0, 0],[0, 3, .1]))

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