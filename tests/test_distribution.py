import cuqi
import numpy as np

from pytest import approx

def test_Normal_mean_standard():
    assert cuqi.distribution.Normal(0,1).mean == approx(0.0)

def test_Normal_pdf_mean():
    pX = cuqi.distribution.Normal(0.1,1)
    assert pX.pdf(0.1) == approx(1.0/np.sqrt(2.0*np.pi))

def test_Normal_sample_regression():
    rng = np.random.RandomState(0) #Replaces legacy method: np.random.seed(0)
    samples = cuqi.distribution.Normal(2,3.5).sample(2,rng=rng)
    target = np.array([[8.17418321], [3.40055023]])
    assert np.allclose( samples, target)

def test_Gaussian_mean():
    mean = np.array([0, 0])
    std = np.array([1, 1])
    R = np.array([[1, -0.7], [-0.7, 1]])
    pX_1 = cuqi.distribution.Gaussian(mean, std, R)
    assert np.allclose(pX_1.mean, np.array([0, 0]) ) 


def test_Gaussian_cov():
    mean = np.array([0, 0])
    std = np.array([1.3, 2])
    R = np.array([[1, -0.7], [-0.7, 1]])
    D = np.diag(std)
    S = D @ R @ D
    pX_1 = cuqi.distribution.Gaussian(mean, std, R)
    assert np.allclose(pX_1.Sigma, S) 

def test_Gaussian_sample_regression():
    rng = np.random.RandomState(0)
    mean = np.array([0, 0])
    std = np.array([1, 1])
    R = np.array([[1, -0.7], [-0.7, 1]])
    pX_1 = cuqi.distribution.Gaussian(mean, std, R)
    samples = pX_1.sample(5,rng=rng)
    target = np.array([[-1.47139568, -0.03445763, -2.10030149, -0.93455864,  0.2541872 ],
       [ 1.78135612,  1.77024604,  1.3433053 ,  0.81731785,  0.06386104]])
    assert np.allclose( samples, target)
    
