import cuqi
import numpy as np

from pytest import approx
import pytest

def test_Normal_mean_standard():
    assert cuqi.distribution.Normal(0,1).mean == approx(0.0)

def test_Normal_pdf_mean():
    pX = cuqi.distribution.Normal(0.1,1)
    assert pX.pdf(0.1) == approx(1.0/np.sqrt(2.0*np.pi))

@pytest.mark.parametrize("mean,var,expected",[
                            (2,3.5,[[8.17418321], [3.40055023]]),
                            (3.141592653589793,2.6457513110645907,[[7.80883646],[4.20030911]]),
                            (-1e-09, 1000000.0,[[1764052.34596766],[400157.20836722]]),
                            (1.7724538509055159, 0, [[1.77245385],[1.77245385]])
                        ])
def test_Normal_sample_regression(mean,var,expected):
    rng = np.random.RandomState(0) #Replaces legacy method: np.random.seed(0)
    samples = cuqi.distribution.Normal(mean,var).sample(2,rng=rng)
    target = np.array(expected)
    assert np.allclose( samples.samples, target)

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

@pytest.mark.parametrize("mean,std,R,expected",[
                        (([0, 0]),
                        ([1, 1]),
                        ([[ 1. , -0.7],
                          [-0.7,  1. ]]),
                        ([[-1.47139568, -0.03445763, -2.10030149, -0.93455864,  0.2541872 ],
                          [ 1.78135612,  1.77024604,  1.3433053 ,  0.81731785,  0.06386104]])),

                        (([-3.14159265,  2.23606798]),
                        ([ 3.14159265, 50.        ]),
                        ([[ 1.   ,  0.001],
                          [-0.001,  1.   ]]),
                        ([[-1.88998123,  3.89532175, -6.21764722, -3.62006864, -1.85133578],
                          [90.43876395, 51.17340778, 95.61377534, 49.74045909, -2.92479388]])),

                        (([23.        ,  0.        ,  3.14159265]),
                        ([3.        , 1.41421356, 3.14159265]),
                        ([[ 1. ,  0.9,  0.3],
                          [0.9,  1. ,  0.5],
                          [-0.3, -0.5,  1. ]]),
                        ([[18.09724511, 16.68961957, 20.05485545, 22.20383097, 20.87158838],
                          [-2.79369395, -1.68366644, -1.31546803, -1.31802001, -1.24045592],
                          [ 0.85185965, -3.99395632,  3.10916299,  2.3865564 ,  2.31523876]]))
                        ])
def test_Gaussian_sample_regression(mean,std,R,expected):
    rng = np.random.RandomState(0)
    pX_1 = cuqi.distribution.Gaussian(np.array(mean), np.array(std), np.array(R))
    samples = pX_1.sample(5,rng=rng).samples
    assert np.allclose( samples, np.array(expected))

@pytest.mark.parametrize("seed",[3,4,5],ids=["seed3","seed4","seed5"])
@pytest.mark.parametrize("mean,var",[(2,3),(np.pi,0),(np.sqrt(5),1)]) 
def test_Normal_rng(mean,var,seed):
    np.random.seed(seed)
    rng = np.random.RandomState(seed)
    assert np.allclose(cuqi.distribution.Normal(mean,var).sample(10).samples,cuqi.distribution.Normal(mean,var).sample(10,rng=rng).samples)

@pytest.mark.parametrize("mean,std,R",[
                        (([0, 0]),
                        ([1, 1]),
                        ([[ 1. , -0.7],
                          [-0.7,  1. ]])),

                        (([-3.14159265,  2.23606798]),
                        ([ 3.14159265, 50.        ]),
                        ([[ 1.   ,  0.001],
                          [-0.001,  1.   ]])),
                        ])
def test_Gaussian_rng(mean,std,R):
    np.random.seed(3)
    rng = np.random.RandomState(3)
    assert np.allclose(cuqi.distribution.Gaussian(mean,std,R).sample(10).samples,cuqi.distribution.Gaussian(mean,std,R).sample(10,rng=rng).samples)

@pytest.mark.parametrize("dist",[cuqi.distribution.GMRF(np.ones(128),35,128,1,'zero'),cuqi.distribution.GMRF(np.ones(128),35,128,1,'periodic'),cuqi.distribution.GMRF(np.ones(128),35,128,1,'neumann')])
def test_GMRF_rng(dist):
    np.random.seed(3)
    rng = np.random.RandomState(3)
    assert np.allclose(dist.sample(10),dist.sample(10,rng=rng))

def test_Uniform_logpdf():
    low = np.array([1, .5])
    high = np.array([2, 2.5])
    UD = cuqi.distribution.Uniform(low, high)
    assert np.allclose(UD.logpdf(np.array([1,.5])), np.log(.5) ) 

def test_Uniform_sample():
    low = np.array([1, .5])
    high = np.array([2, 2.5])
    rng = np.random.RandomState(3)
    UD = cuqi.distribution.Uniform(low, high)
    cuqi_samples = UD.sample(3,rng=rng)
    assert np.allclose(cuqi_samples.samples,\
                       np.array([[1.5507979 , 1.29090474, 1.89294695],
                       [1.91629565, 1.52165521, 2.29258618]]) ) 
     