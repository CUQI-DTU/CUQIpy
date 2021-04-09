import cuqi
import numpy as np

import pytest
from pytest import approx

@pytest.mark.parametrize("dim,kernel,kernel_param,expected",[
                            (128,"Gauss",None,45.3148),
                            (36,"Gauss",None,12.7448),
                            (36,"Gauss",5,18.0238),
                        ])
def test_Deconvolution_MatrixNorm_regression(dim,kernel,kernel_param,expected):
    tp = cuqi.testproblem.Deconvolution(dim=dim,kernel=kernel,kernel_param=kernel_param)
    assert np.linalg.norm(tp.model.get_matrix()) == approx(expected,rel=1e-4)

@pytest.mark.parametrize("dim,phantom,phantom_param,expected",[
                            (36,"Gauss",None,2.0944),
                            (36,"Gauss",3,2.7039),
                            (128,"Gauss",10,2.8211),
                        ])
def test_Deconvolution_PhantomNorm_regression(dim,phantom,phantom_param,expected):
    tp = cuqi.testproblem.Deconvolution(dim=dim,phantom=phantom,phantom_param=phantom_param)
    assert np.linalg.norm(tp.exactSolution) == approx(expected,rel=1e-4)