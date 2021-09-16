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

def test_initialize_model_geom():
    range_geometry = cuqi.geometry.Continuous1D(shape=[5])
    domain_geometry = cuqi.geometry.Continuous1D(shape=3)
    model1 = cuqi.model.Model(lambda x:x,range_geometry, domain_geometry)
    dims_old = model1.dims 
    model1.range_geometry = cuqi.geometry.Continuous1D(shape=[4])
    assert(dims_old == (5,3) and model1.dims == (4,3)) 

def test_initialize_model_matr():
    model1 = cuqi.model.LinearModel(np.eye(5))
    assert( model1.dims == (5,5) and model1.domain_geometry.shape == 5 and
            len(model1.range_geometry.grid) == 5)
