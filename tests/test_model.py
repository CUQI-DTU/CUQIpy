import numpy as np
import scipy as sp
import cuqi
import pytest


@pytest.mark.parametrize("seed",[(0),(1),(2)])
def test_LinearModel_getMatrix(seed):
    np.random.seed(seed)
    A = np.random.randn(10,7) #Random matrix

    model1 = cuqi.model.LinearModel(A)
    model2 = cuqi.model.LinearModel(lambda x : A@x, lambda y: A.T@y, range=A.shape[0], domain=A.shape[1])
    
    mat1 = model1.get_matrix() #Normal matrix
    mat2 = model2.get_matrix() #Sparse matrix (generated from functions)

    assert np.allclose(mat1,mat2.A)

def test_initialize_model_dim():
    model1 = cuqi.model.Model(lambda x:x, range=4, domain=4)
    assert(len(model1.domainGeom.grid) == 4 and len(model1.rangeGeom.grid) == 4 )

def test_initialize_model_geom():
    range = cuqi.geometry.Continuous1D(dim=[5])
    domain = cuqi.geometry.Continuous1D(dim=[3])
    model1 = cuqi.model.Model(lambda x:x,range, domain)
    dim_old = model1.dim 
    model1.rangeGeom = cuqi.geometry.Continuous1D(dim=[4])
    assert(dim_old == (5,3) and model1.dim == (4,3)) 

def test_initialize_model_matr():
    model1 = cuqi.model.LinearModel(np.eye(5))
    assert( model1.dim == (5,5) and len(model1.domainGeom.grid) == 5 and
            len(model1.rangeGeom.grid) == 5)
