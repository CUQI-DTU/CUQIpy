import numpy as np
import scipy as sp
import cuqi
import pytest


@pytest.mark.parametrize("seed",[(0),(1),(2)])
def test_LinearModel_getMatrix(seed):
    np.random.seed(seed)
    A = np.random.randn(10,7) #Random matrix

    model1 = cuqi.model.LinearModel(A)
    model2 = cuqi.model.LinearModel(lambda x : A@x, lambda y: A.T@y, dim=A.shape)
    
    mat1 = model1.get_matrix() #Normal matrix
    mat2 = model2.get_matrix() #Sparse matrix (generated from functions)

    assert np.allclose(mat1,mat2.A)
