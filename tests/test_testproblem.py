import cuqi
import numpy as np

import pytest
from pytest import approx
from scipy.ndimage import convolve

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
    tp = cuqi.testproblem.Deconvolution1D(dim=dim, PSF=kernel, PSF_param=kernel_param, use_legacy=True)
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
    tp = cuqi.testproblem.Deconvolution1D(dim=dim,phantom=phantom,phantom_param=phantom_param)
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
    ("scaledgaussian",0.2,74.5626529810876),   
    ("scaledgaussian",3,235.68073267350434),   
    ("gaussian",0.2,75.7589097857315),  
    ("gaussian",3,79.76620602993175),  
])
def test_Deconvolution_noise_regression(noise_type,noise_std,expected):
    np.random.seed(1337)
    tp = cuqi.testproblem.Deconvolution1D(noise_type=noise_type,noise_std=noise_std, use_legacy=True, phantom="gauss")
    assert np.linalg.norm(tp.data) == approx(expected)

def test_Deconvolution_custom_kernel():
    """ Tests using a custom kernel. """
    grid = np.linspace(-1, 1, 128)
    kernel = np.exp(-(15*grid)**2)

    TP = cuqi.testproblem.Deconvolution1D(dim=128, PSF=kernel, phantom="pc", use_legacy=True, noise_std=0.05)

    # Test expected value
    assert np.linalg.norm(TP.exactData) == pytest.approx(83.40353338231859)

    # Test raising of error if kernel is not 1D
    with pytest.raises(ValueError, match=r"PSF must be a 1D array"):
        TP = cuqi.testproblem.Deconvolution1D(dim=128, PSF=np.ones((128, 128)))


def test_Deconvolution_custom_phantom():
    """ Tests using a custom phantom. """
    grid = np.linspace(-1, 1, 128)
    phantom = np.sin(grid)

    TP = cuqi.testproblem.Deconvolution1D(dim=128, phantom=phantom)

    # Test expected value
    assert np.linalg.norm(TP.exactSolution) == approx(5.94476801594791)

    # Test raising of error if phantom is not 1D
    with pytest.raises(ValueError, match=r"phantom must be a 1D array"):
        TP = cuqi.testproblem.Deconvolution1D(dim=128, phantom=np.ones((128, 128)))

    # Test raising of error if phantom is not same length as dim
    with pytest.raises(ValueError, match=r"phantom must be a 1D array of length dim"):
        TP = cuqi.testproblem.Deconvolution1D(dim=128, phantom=np.ones(125))


@pytest.mark.parametrize("model_class", [cuqi.testproblem.Poisson1D, cuqi.testproblem.Heat1D])
@pytest.mark.parametrize("observation_map, expected_grid_equal",
                         [(None, True),
                          (lambda x: x[np.where(x > np.pi/2)], False)])
def test_testproblem_pde_grid_obs(model_class, observation_map, expected_grid_equal):
    """ Test that the observation operator works on grids or not. """
    N = 128
    L = np.pi

    model = model_class(dim=N, endpoint=L, field_type="Step",
                        observation_grid_map=observation_map).model

    assert model.pde.grids_equal == expected_grid_equal

def test_Poisson():
    # %% Poisson
    N = 129 # spatial discretization 
    L = np.pi   # Length of domain
    amp_fact = 10
    f = lambda xs: 10*np.exp( -( (xs - 0.5)**2 ) / 0.02)
    map = lambda x: np.exp(amp_fact*x)
    dx = L/(N-1)
    x = np.linspace(dx/2,L-dx/2,N)
    true_kappa = np.exp( 5*x*np.exp(-2*x)*np.sin(L-x) )
    model = cuqi.testproblem.Poisson1D(dim=N, endpoint=L, source=f, field_type="KL", map=map).model

    assert np.linalg.norm(model.forward(true_kappa, is_par=False)) == approx(6.183419642601269)
    assert np.linalg.norm(model.forward(np.ones(model.domain_dim))) == approx(5.195849938761418)

def test_Heat():
    # %% HEAT
    N = 128
    L = np.pi
    dx = L/(N+1)
    x = np.linspace(dx,L,N,endpoint=False)
    true_kappa = x*np.exp(-2*x)*np.sin(L-x)
    model = cuqi.testproblem.Heat1D(dim=N, endpoint=L, field_type="KL").model

    assert np.linalg.norm(model.forward(true_kappa, is_par=False)) == approx(0.5476375563249378)
    assert np.linalg.norm(model.forward(np.ones(model.domain_dim))) == approx(0.548657107830281)

def test_Abel():
    N = 128
    L = 1
    h = L/N
    tvec = np.linspace(h/2, L-h/2, N)
    true_image = np.sin(tvec*np.pi)*np.exp(-2*tvec)
    KL_map = lambda x: 10*x
    model = cuqi.testproblem.Abel1D(dim=N, endpoint=L, field_type="KL", KL_map=KL_map).model

    assert np.linalg.norm(model.forward(true_image, is_par=False)) == approx(4.580752014966276)
    assert np.linalg.norm(model.forward(np.ones(model.domain_dim))) == approx(9.37456136258068)

#Deconv 2D tests
#TODO. Add tests for custom PSF
@pytest.mark.parametrize("prior",[
    (cuqi.distribution.Gaussian(np.zeros(128**2), 1, name="x")),
    #(cuqi.distribution.LMRF(0, 1, "zeros", geometry=128**2)),
    #(cuqi.distribution.CMRF(np.zeros(128**2), 1, "zeros")),
])
def test_Deconvolution2D_Sampling_prior(prior): 
    tp = cuqi.testproblem.Deconvolution2D(prior=prior)
    tp.prior.geometry = tp.model.domain_geometry
    tp.sample_posterior(10) # Tests that sampling at least runs withour error.
    #TODO. Make into regression tests + other samplers. Move some tests to BayesianProblem

@pytest.mark.parametrize("phantom",[
    ( np.zeros((128,128)) ),    #Correct size
    ( np.zeros(128**2) ),       #Correct size, but vector
    ( np.zeros((64,64)) ),      #Different size. Should get reshaped
    ( np.zeros(64**2) ),        #Different size. Should get reshaped
    ( "satellite" ),            #String for specific cases
])
def test_Deconvolution2D_phantom(phantom):
    TP= cuqi.testproblem.Deconvolution2D(dim=128, phantom=phantom)
    assert TP.exactSolution.shape == (128**2,)

@pytest.mark.parametrize("PSF_size",[20, 21])
@pytest.mark.parametrize("PSF",["Gauss", "Moffat", "Defocus"])
@pytest.mark.parametrize("BC_MAP",[ 
    ["periodic", "wrap"],
    ["zero", "constant"],
    ["mirror", "mirror"],
    ["Neumann", "reflect"],
    ["nearest", "nearest"],
    ])
def test_Deconvolution2D_convolve(BC_MAP, PSF, PSF_size):
    """ Compare Convolve with ndimage slow convolve"""
    # Define testproblem
    TP = cuqi.testproblem.Deconvolution2D(BC=BC_MAP[0], PSF=PSF, PSF_size=PSF_size)

    # Extract kernel + image + data
    P = TP.Miscellaneous["PSF"]
    x = TP.exactSolution.funvals # Image
    y = TP.exactData.funvals # Data

    # ndimage convolve forward and adjoint (non-FFT)
    forward = lambda x: convolve(x, P, mode=BC_MAP[1])
    adjoint = lambda y: convolve(y, np.flipud(np.fliplr(P)), mode=BC_MAP[1])

    # Compare forward
    assert np.allclose(forward(x), TP.model.forward(x).funvals)
    
    # Compare adjoint
    assert np.allclose(adjoint(y), TP.model.adjoint(y).funvals)

def test_WangCubic():
    """ Test the WangCubic testproblem with simple regression test"""
    TP = cuqi.testproblem.WangCubic()
    assert TP.posterior.logd([0.7, 1.3]) == approx(-119.4710156)
    assert np.allclose(TP.posterior.gradient([0.7, 1.3]), np.array([[26.174, -153.5]]))

    