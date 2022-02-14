import cuqi
import numpy as np

import pytest
from pytest import approx

@pytest.mark.parametrize("dim,kernel,kernel_param,expected",[
    (128,"Gauss",None,45.3148),
    (36,"Gauss",None,12.7448),
    (36,"Gauss",5,18.0238),
    (36,"sinc",None,9.2320),
    (36,"sinc",37,6.0585),
    (36,"prolate",None,9.2320),
    (36,"prolate",37,6.0585),
    (36,"vonmises",None,12.8714),
    (36,"vonmises",37,7.7606),
])
def test_Deconvolution_MatrixNorm_regression(dim,kernel,kernel_param,expected):
    tp = cuqi.testproblem.Deconvolution(dim=dim,kernel=kernel,kernel_param=kernel_param)
    assert np.linalg.norm(tp.model.get_matrix()) == approx(expected,rel=1e-4)

@pytest.mark.parametrize("dim,phantom,phantom_param,expected",[
    (36,"Gauss",None,2.0944),
    (36,"Gauss",3,2.7039),
    (128,"Gauss",10,2.8211),
    (36,"sinc",None,1.8518),
    (36,"vonmises",None,2.1582),
    (36,"square",None,2),
    (36,"hat",None,1.2247),
    (36,"bumps",None,2.4009),
    (36,"derivgauss",None,2.4858),                            
])
def test_Deconvolution_PhantomNorm_regression(dim,phantom,phantom_param,expected):
    tp = cuqi.testproblem.Deconvolution(dim=dim,phantom=phantom,phantom_param=phantom_param)
    assert np.linalg.norm(tp.exactSolution) == approx(expected,rel=1e-4)

@pytest.mark.parametrize("dim,phantom,phantom_param,expected",[
    (35,"gauss",None,2.0643),   
    (35,"sinc",None,1.8252),
    (35,"vonmises",None,2.0848),
    (35,"square",None,2),
    (35,"hat",None,1.2247),
    (35,"bumps",None,2.3673),
    (35,"derivgauss",None,2.5478),  
])
def test_Deconvolution_PhantomNorm2_regression(dim,phantom,phantom_param,expected):
    x = cuqi.testproblem._getExactSolution(dim=dim,phantom=phantom,phantom_param=phantom_param)
    assert np.linalg.norm(x) == approx(expected,rel=1e-4)

@pytest.mark.parametrize("noise_type,noise_std,expected",[
    ("scaledgaussian",0.2,74.32896068980779),   
    ("scaledgaussian",3,249.12043011440917),   
    ("gaussian",0.2,75.7589097857315),  
    ("gaussian",3,79.76620602993175),  
])
def test_Deconvolution_noise_regression(noise_type,noise_std,expected):
    np.random.seed(1337)
    tp = cuqi.testproblem.Deconvolution(noise_type=noise_type,noise_std=noise_std)
    assert np.linalg.norm(tp.data) == approx(expected)

# Test the observation operator working on grids or not
def test_testproblem_pde_grid_obs():
    N = 128
    L = np.pi
    T = 0.2

    model1 = cuqi.testproblem.Poisson_1D(dim=N,endpoint=L, field_type="Step").model
    model2 = cuqi.testproblem.Poisson_1D(dim=N,dim_obs=10,endpoint=L, field_type="Step").model
    model3 = cuqi.testproblem.Heat_1D(dim=N,endpoint=L, field_type="Step").model
    model4 = cuqi.testproblem.Heat_1D(dim=N,dim_obs=10,endpoint=L, field_type="Step").model

    assert model1.pde.grids_equal == True
    assert model2.pde.grids_equal == False
    assert model3.pde.grids_equal == True
    assert model4.pde.grids_equal == False

def test_Poisson():
    # %% Poisson
    N = 129 # spatial discretization 
    L = np.pi   # Length of domain
    amp_fact = 10
    f = lambda xs: 10*np.exp( -( (xs - 0.5)**2 ) / 0.02)
    KL_map = lambda x: np.exp(amp_fact*x)
    dx = L/(N-1)
    x = np.linspace(dx/2,L-dx/2,N)
    true_kappa = np.exp( 5*x*np.exp(-2*x)*np.sin(L-x) )
    model = cuqi.testproblem.Poisson_1D(dim=N, endpoint=L, source=f, field_type="KL", KL_map=KL_map).model

    assert np.linalg.norm(model.forward(true_kappa, is_par=False)) == approx(6.183419642601269)
    assert np.linalg.norm(model.forward(np.ones(model.domain_dim))) == approx(5.195849938761418)

def test_Heat():
    # %% HEAT
    N = 128
    L = np.pi
    dx = L/(N+1)
    x = np.linspace(dx,L,N,endpoint=False)
    true_kappa = x*np.exp(-2*x)*np.sin(L-x)
    model = cuqi.testproblem.Heat_1D(dim=N, endpoint=L, field_type="KL").model

    assert np.linalg.norm(model.forward(true_kappa, is_par=False)) == approx(0.5478069279144476)
    assert np.linalg.norm(model.forward(np.ones(model.domain_dim))) == approx(0.5487907807357283)

def test_Abel():
    N = 128
    L = 1
    h = L/N
    tvec = np.linspace(h/2, L-h/2, N)
    true_image = np.sin(tvec*np.pi)*np.exp(-2*tvec)
    KL_map = lambda x: 10*x
    model = cuqi.testproblem.Abel_1D(dim=N, endpoint=L, field_type="KL", KL_map=KL_map).model

    assert np.linalg.norm(model.forward(true_image, is_par=False)) == approx(4.580752014966276)
    assert np.linalg.norm(model.forward(np.ones(model.domain_dim))) == approx(9.37456136258068)
