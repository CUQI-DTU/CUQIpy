import cuqi
import numpy as np

from pytest import approx

def test_Normal_mean_standard():
    assert cuqi.distribution.Normal(0,1).mean == approx(0.0)

def test_Normal_pdf_mean():
    pX = cuqi.distribution.Normal(0.1,1)
    assert pX.pdf(0.1) == approx(1.0/np.sqrt(2.0*np.pi))

def test_Normal_sample_regression():
    np.random.seed(0)
    samples = cuqi.distribution.Normal(2,3.5).sample(2)
    target = np.array([[8.17418321], [3.40055023]])
    assert np.allclose( samples, target)

def test_Gaussian():
    mean = np.array([0, 0])
    std = np.array([1, 1])
    R = np.array([[1, -0.7], [-0.7, 1]])
    pX_1 = cuqi.distribution.Gaussian(mean, std, R)
    assert np.allclose(pX_1.mean, np.array([0, 0]) ) 
