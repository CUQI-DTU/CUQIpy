from math import isnan
import cuqi
import numpy as np
import scipy as sp
import scipy.stats as scipy_stats
import scipy.sparse as sps
import numpy.linalg as nplinalg

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
    target = np.array(expected).T
    assert np.allclose( samples.samples, target)

@pytest.mark.parametrize("mean,std,points,expected",[
                            (0,1,[-1,0,1],[1,0,-1]),
   (np.array([0,0]),np.array([1,1]),[[-1,0],[0,0],[0,-1]], [[1,0],[0,0],[0,1]])])
def test_Normal_gradient(mean,std,points,expected):
    p = cuqi.distribution.Normal(mean,std)
    for point, grad in zip(points, expected):
        assert np.allclose(p.gradient(point), grad)

@pytest.mark.parametrize("mean,std,low,high,points",[(0.0, 
                                               1.0,
                                               -1.0,
                                               1.0,
                                               [-1.5, -0.5, 0.5, 1.5]),
                                              (np.array([0.0, 0.0]), 
                                               np.array([1.0, 1.0]),
                                               np.array([-1.0, -1.0]),
                                               np.array([1.0, 1.0]),
                                               [np.array([-0.5, 0.0]),
                                                np.array([0.5, 0.0]),
                                                np.array([-2.0, 0.0]),
                                                np.array([2.0, 0.0])])])
def test_TruncatedNormal_logpdf(mean,std,low,high,points):
    x_trun = cuqi.distribution.TruncatedNormal(mean,std,low=low,high=high)
    x = cuqi.distribution.Normal(mean,std)
    for point in points:
        if np.all(point >= low) and np.all(point <= high):
            assert x_trun.logpdf(point) == approx(x.logpdf(point))
        else:
            assert np.isneginf(x_trun.logpdf(point))

@pytest.mark.parametrize("mean,std,low,high,points",[(0.0, 
                                               1.0,
                                               -1.0,
                                               1.0,
                                               [-1.5, -0.5, 0.5, 1.5]),
                                              (np.array([0.0, 0.0]), 
                                               np.array([1.0, 1.0]),
                                               np.array([-1.0, -1.0]),
                                               np.array([1.0, 1.0]),
                                               [np.array([-0.5, 0.0]),
                                                np.array([0.5, 0.0]),
                                                np.array([-2.0, 0.0]),
                                                np.array([2.0, 0.0])])])
def test_TruncatedNormal_gradient(mean,std,low,high,points):
    x_trun = cuqi.distribution.TruncatedNormal(mean,std,low=low,high=high)
    x = cuqi.distribution.Normal(mean,std)
    for point in points:
        if np.all(point >= low) and np.all(point <= high):
            assert np.all(x_trun.gradient(point) == approx(x.gradient(point)))
        else:
            assert np.all(np.isnan(x_trun.gradient(point)))

@pytest.mark.parametrize("mean,std,low,high",[(0.0, 
                                               1.0,
                                               -1.0,
                                               1.0),
                                              (np.array([0.0, 0.0]), 
                                               np.array([1.0, 1.0]),
                                               np.array([-1.0, -1.0]),
                                               np.array([1.0, 1.0]))])
def test_TruncatedNormal_sampling(mean,std,low,high):
    x = cuqi.distribution.TruncatedNormal(mean,std,low=low,high=high)
    samples = x.sample(10000).samples
    for i in range(samples.shape[1]):
        sample = samples[:,i]
        assert np.all(sample >= low) and np.all(sample <= high)

def test_Gaussian_mean():
    mean = np.array([0, 0])
    std = np.array([1, 1])
    R = np.array([[1, -0.7], [-0.7, 1]])
    cov = np.diag(std) @ (R @ np.diag(std))
    pX_1 = cuqi.distribution.Gaussian(mean, cov)
    assert np.allclose(pX_1.mean, np.array([0, 0]) ) 

def test_Gaussian_cov():
    mean = np.array([0, 0])
    std = np.array([1.3, 2])
    R = np.array([[1, -0.7], [-0.7, 1]])
    D = np.diag(std)
    S = D @ R @ D
    pX_1 = cuqi.distribution.Gaussian(mean, S)
    assert np.allclose(pX_1.cov, S)
    
def test_Gaussian_multiple():
    pX_1 = cuqi.distribution.Gaussian(np.zeros(2), np.array([[1.5, -.5],[-.5, 1]]))
    #
    X, Y = np.meshgrid(np.linspace(-4, 4, 200), np.linspace(-4, 4, 200))
    Xf, Yf = X.flatten(), Y.flatten()
    pts = np.vstack([Xf, Yf]).T   # pts is (m*n, d)
    npoints = pts.shape[0]
    Z = pX_1.pdf(pts)
    assert npoints == Z.shape[0]

@pytest.mark.xfail(reason="Expected to fail after fixing Gaussian sample. Regression needs to be updated")
@pytest.mark.parametrize("mean,std,R,expected",[
                        (([0, 0]),
                        ([1, 1]),
                        ([[ 1.0 , -0.7],
                          [-0.7,  1. ]]),
                        ([[ 0.30471644,  0.52294788,  0.3204432 ,  0.82791771,  0.88232622],
                          [-2.52738159,  0.50701152, -1.04189629, -2.16116453, -1.34325028]])),

                        (([-3.14159265,  2.23606798]),
                        ([ 3.14159265, 50.        ]),
                        ([[ 1.0   ,  -0.001],
                          [-0.001,  1.0   ]]),
                        ([[  2.40014477,  -1.88427406,  -0.06682814,   3.89835695,  2.72559222],
                          [-46.63338991,  49.73922674,  -5.33487942,  -2.93194249, 22.76010272]])),

                        (([23.        ,  0.0        ,  3.14159265]),
                        ([3.0        , 1.41421356, 3.14159265]),
                        ([[ 1.0, 0.9,  0.3],
                          [0.9,  1. , -0.5],
                          [0.3, -0.5,  1.0 ]]),
                        ([[23.7305748 , 22.58046919, 23.19981226, 23.43140839, 23.1244483 ],
                          [ 2.83554239,  3.92371224,  3.06634238,  4.00865492,  4.22990048],
                          [ 0.39008306,  4.56363292,  2.81075259, -1.94308617, -0.87372503]]))
                        ])
def test_Gaussian_sample_regression(mean,std,R,expected):
    rng = np.random.RandomState(0)
    std = np.array(std)
    R = np.array(R)
    cov = np.diag(std) @ (R @ np.diag(std))
    pX_1 = cuqi.distribution.Gaussian(np.array(mean), cov)
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
                        ([[ 1.   ,  -0.001],
                          [-0.001,  1.   ]])),
                        ])
def test_Gaussian_rng(mean,std,R):
    np.random.seed(3)
    rng = np.random.RandomState(3)
    std = np.array(std)
    R = np.array(R)
    cov = np.diag(std) @ (R @ np.diag(std))
    assert np.allclose(cuqi.distribution.Gaussian(mean,cov).sample(10).samples,cuqi.distribution.Gaussian(mean,cov).sample(10,rng=rng).samples)

@pytest.mark.parametrize("dist",[cuqi.distribution.GMRF(np.ones(128),35,'zero'),cuqi.distribution.GMRF(np.ones(128),35,'periodic'),cuqi.distribution.GMRF(np.ones(128),35,'neumann')])
def test_GMRF_rng(dist):
    np.random.seed(3)
    rng = np.random.RandomState(3)
    assert np.allclose(dist.sample(10).samples,dist.sample(10,rng=rng).samples)

@pytest.mark.parametrize( \
  "low,high,toeval,expected",[ \
    (-2.0, 3.0, 1.0, np.log(0.2)), \
    (-2.0, 3.0, 3.5, -np.inf), \
    (np.array([1.0, 0.5]), np.array([2.0, 2.5]), np.array([1, 0.5]), np.log(0.5)), \
    (np.array([1.0, 0.5]), np.array([2.0, 2.5]), np.array([0.0, 0.5]), -np.inf), \
    (np.array([1.0, 0.5]), np.array([2.0, 2.5]), np.array([0.0, 0.0]), -np.inf), \
    (np.array([1.0, 0.5]), np.array([2.0, 2.5]), np.array([3.0, 0.5]), -np.inf), \
    (np.array([1.0, 0.5]), np.array([2.0, 2.5]), np.array([3.0, 3.0]), -np.inf), \
    (np.array([1.0, 2.0, 3.0]), np.array([3.0, 4.0, 5.0]), np.array([3.0, 4.0, 3.0]), np.log(0.125)), \
    (np.array([1.0, 2.0, 3.0]), np.array([3.0, 4.0, 5.0]), np.array([0.0, 5.0, 4.0]), -np.inf) \
  ])
def test_Uniform_logpdf(low, high, toeval, expected):
    UD = cuqi.distribution.Uniform(low, high)
    assert np.allclose(UD.logpdf(toeval), expected) 


@pytest.mark.parametrize("low,high,expected",[(np.array([1, .5]), 
                                               np.array([2, 2.5]),
                                               np.array([[1.5507979 , 1.29090474, 1.89294695],
                                               [1.91629565, 1.52165521, 2.29258618]])),
                                              (1,2,
                                               np.array([[1.5507979 , 1.70814782, 1.29090474]]))])
def test_Uniform_sample(low, high, expected):
    rng = np.random.RandomState(3)
    UD = cuqi.distribution.Uniform(low, high)
    cuqi_samples = UD.sample(3,rng=rng)
    print(cuqi_samples)
    assert np.allclose(cuqi_samples.samples, expected) 

@pytest.mark.parametrize("distribution, kwargs",
                         [(cuqi.distribution.Uniform, 
                          {'low':np.array([2, 2.5, 3,5]),
                          'high':np.array([5,7, 7,6])}),
                          (cuqi.distribution.Gaussian, 
                          {'mean':np.array([0, 0, 0, 0]),
                          'sqrtcov':np.array([1, 1, 1, 1])})
                          ])
def test_distribution_contains_geometry(distribution, kwargs):
    rng = np.random.RandomState(3)
    geom = cuqi.geometry.Continuous2D((2,2))
    dist = distribution(**kwargs,geometry = geom)
    cuqi_samples = dist.sample(3,rng=rng)
    assert(dist.dim == geom.par_dim and 
          cuqi_samples.geometry == geom and
          dist.geometry == geom and 
          np.all(geom.grid[0]==np.array([0, 1])) and
          np.all(geom.grid[1]== np.array([0, 1])))

# Compare computed covariance
@pytest.mark.parametrize("seed", [0,3,5,7])
@pytest.mark.parametrize("mean,cov,mean_full,cov_full",[
    ( (0),           (5),            (0),           (5)           ),
    ( (0),           (5*np.ones(3)), (np.zeros(3)), (5*np.eye(3)) ),
    ( (np.zeros(3)), (5),            (np.zeros(3)), (5*np.eye(3)) ),
    ( (np.zeros(3)), (5*np.ones(3)), (np.zeros(3)), (5*np.eye(3)) ),
    ( (0),           (5*np.eye(3)),  (np.zeros(3)), (5*np.eye(3)) ),
    ( (0),           (5*sps.eye(3)), (np.zeros(3)), (5*np.eye(3)) ),
    ( (0), (np.array([[5,-3],[-3,2]])),       (np.zeros(2)), (np.array([[5,-3],[-3,2]])) ),
    #( (0), (sps.csc_matrix([[5,3],[-3,2]])), (np.zeros(2)), (np.array([[5,3],[-3,2]])) ),
])
def test_Gaussian_cov(mean,cov, mean_full, cov_full, seed):
    #Fix seed for reproducibility
    np.random.seed(seed)

    # Define cuqi dist using various means and covs
    prior = cuqi.distribution.Gaussian(mean, cov)

    # Compare logpdf with scipy using full vector+matrix rep
    x0 = 1000*np.random.rand(prior.dim)
    eval1 = prior.logpdf(x0)
    eval2 = sp.stats.multivariate_normal.logpdf(x0, mean_full, cov_full)

    assert np.allclose(eval1,eval2)

    gradeval1 = prior.gradient(x0)
    gradeval2 = sp.optimize.approx_fprime(x0, prior.logpdf, 1e-15)

    assert np.allclose(gradeval1,gradeval2)


def _stats(samples):
    """ Compute meadian, std, and lo95, up95 of samples """
    return np.vstack((np.median(samples, axis=1), np.std(samples, axis=1), np.percentile(samples, 2.5, axis=1), np.percentile(samples, 97.5, axis=1)))

@pytest.mark.parametrize("prec, GMRF_order",[
    (sps.diags([-1, 2, -1], [-1, 0, 1], shape=(5, 5)), 1),
    #(sps.diags([-1, 2, -1], [-1, 0, 1], shape=(128, 128)), 1),
    (sps.diags([1, -4, 6, -4, 1], [-2, -1, 0, 1, 2], shape=(5, 5)), 2),
    #(sps.diags([1, -4, 6, -4, 1], [-2, -1, 0, 1, 2], shape=(128, 128)), 2),
    (sps.eye(5), 0),
])
def test_Gaussians_vs_GMRF(prec, GMRF_order):
    """ Tests the various Gaussians given some common precision matrices related to GMRFs
    
    This tests both logpdf, gradient, and sample methods for sparse and dense matricies as input.
    """

    # Get dimension of precision matrix
    dim = prec.shape[0]

    # Store normal and sparse version of precision matrix
    prec_s = prec
    prec = prec.toarray()

    # Compute covariance from precision
    cov = sp.linalg.inv(prec)
    cov_s = sps.linalg.inv(prec_s)

    # Compute lower triangular Cholesky decomposition of covariance
    sqrtcov = sp.linalg.cholesky(cov, lower=True)
    sqrtcov_s = sp.sparse.csr_matrix(sqrtcov)  

    # Compute upper triangular cholesky decomposition of precision
    sqrtprec = sp.linalg.cholesky(prec)
    sqrtprec_s = sp.sparse.csr_matrix(sqrtprec)    

    # Define Gaussians from all combinations
    X_prec = cuqi.distribution.Gaussian(np.zeros(dim), prec=prec)
    X_cov = cuqi.distribution.Gaussian(np.zeros(dim), cov)
    X_sqrtprec = cuqi.distribution.Gaussian(np.zeros(dim), sqrtprec=sqrtprec)
    X_sqrtcov = cuqi.distribution.Gaussian(np.zeros(dim), sqrtcov=sqrtcov)
    X_GMRF = cuqi.distribution.GMRF(np.zeros(dim), 1, 'zero', order=GMRF_order)

    # logpdfs for full matrix
    x0 = np.random.randn(dim)
    assert np.allclose(X_cov.logpdf(x0), X_GMRF.logpdf(x0))
    assert np.allclose(X_cov.logpdf(x0), X_sqrtprec.logpdf(x0))
    assert np.allclose(X_cov.logpdf(x0), X_prec.logpdf(x0))
    assert np.allclose(X_cov.logpdf(x0), X_sqrtcov.logpdf(x0))

    # gradients for full matrix
    assert np.allclose(X_cov.gradient(x0), X_GMRF.gradient(x0))
    #assert np.allclose(X_cov.gradient(x0), X_sqrtprec.gradient(x0)) #TODO: NotImplementedError
    assert np.allclose(X_cov.gradient(x0), X_prec.gradient(x0))

    # samples (compare statistics)
    Ns = 10000
    s_cov = _stats(X_cov.sample(Ns).samples)
    s_GMRF = _stats(X_GMRF.sample(Ns).samples)
    s_sqrtprec = _stats(X_sqrtprec.sample(Ns).samples)
    s_prec = _stats(X_prec.sample(Ns).samples)
    s_sqrtcov = _stats(X_sqrtcov.sample(Ns).samples)

    # We round to one decimal and allow 10% error in sample statistics.
    # TODO. Better comparrison here..
    # We want to get a precision of e.g. 2 in scientific notation
    # float(np.format_float_scientific(2.367e-01, precision=1) = 0.24
    # float(np.format_float_scientific(9.650e+01, precision=1)) = 96.0
    # its not the same as np.round().
    assert np.allclose(np.round(s_cov, 1), np.round(s_GMRF, 1) , rtol=0.1)
    assert np.allclose(np.round(s_cov, 1), np.round(s_sqrtprec, 1) , rtol=0.1)
    assert np.allclose(np.round(s_cov, 1), np.round(s_prec, 1) , rtol=0.1)
    assert np.allclose(np.round(s_cov, 1), np.round(s_sqrtcov, 1) , rtol=0.1)

    # Now compare sparse versions
    # Check un-normalized logpdfs for sparse precision and covariance
    X_prec_s = cuqi.distribution.Gaussian(np.zeros(dim), prec=prec_s)
    X_cov_s = cuqi.distribution.Gaussian(np.zeros(dim), cov=cov_s)
    X_sqrtprec_s = cuqi.distribution.Gaussian(np.zeros(dim), sqrtprec=sqrtprec_s)
    X_sqrtcov_s = cuqi.distribution.Gaussian(np.zeros(dim), sqrtcov=sqrtcov_s)

    assert np.allclose(X_cov._logupdf(x0), X_cov_s._logupdf(x0))
    assert np.allclose(X_cov._logupdf(x0), X_prec_s._logupdf(x0))
    assert np.allclose(X_cov._logupdf(x0), X_sqrtprec_s._logupdf(x0))
    assert np.allclose(X_cov._logupdf(x0), X_sqrtcov_s._logupdf(x0))

    # Check gradients for sparse precision and covariance
    assert np.allclose(X_cov.gradient(x0), X_cov_s.gradient(x0))
    assert np.allclose(X_cov.gradient(x0), X_prec_s.gradient(x0))
    #assert np.allclose(X_cov.gradient(x0), X_sqrtprec_s.gradient(x0)) # TODO: NotImplementedError
    assert np.allclose(X_cov.gradient(x0), X_sqrtcov_s.gradient(x0))

    # Check samples for sparse precision and covariance
    s_cov_s = _stats(X_cov_s.sample(Ns).samples)
    s_prec_s = _stats(X_prec_s.sample(Ns).samples)
    s_sqrtprec_s = _stats(X_sqrtprec_s.sample(Ns).samples)
    s_sqrtcov_s = _stats(X_sqrtcov_s.sample(Ns).samples)

    assert np.allclose(np.round(s_cov, 1), np.round(s_cov_s, 1) , rtol=0.1)
    assert np.allclose(np.round(s_cov, 1), np.round(s_prec_s, 1) , rtol=0.1)
    assert np.allclose(np.round(s_cov, 1), np.round(s_sqrtprec_s, 1) , rtol=0.1)
    assert np.allclose(np.round(s_cov, 1), np.round(s_sqrtcov_s, 1) , rtol=0.1)
    
    #TODO. Add comparrison of sampling using X_cov.sqrtprec directly. This is what LinearRTO uses.
    # CUQI test problem
    # TP = cuqi.testproblem.Deconvolution1D(dim=n)
    # TP.prior = X_GMRF
    # samples_GMRF = TP._sampleLinearRTO(2000)
    # TP.prior = X_cov
    # samples_cov = TP._sampleLinearRTO(2000)
    # TP.prior = X_sqrtprec
    # samples_sqrtprec = TP._sampleLinearRTO(2000)
    # samples_GMRF.plot_ci()
    # samples_sqrtprec.plot_ci()
    # samples_cov.plot_ci()

def test_InverseGamma_sample():
    a = [1,2]
    location = 0
    scale = 1
    N = 1000
    rng1 = np.random.RandomState(1)
    rng2 = np.random.RandomState(1)
    x = cuqi.distribution.InverseGamma(shape=a, location=location, scale=scale)
    samples1 = x.sample(N, rng=rng1).samples
    samples2 = sp.stats.invgamma.rvs(a=a, loc=location, scale=scale, size=(N,len(a)), random_state=rng2).T

    assert x.dim == len(a)
    assert np.all(np.isclose(samples1, samples2))

@pytest.mark.parametrize("a, location, scale",
                         [([1,2,1], [0,-1, 100], np.array([.1, 1, 20])),
                          (np.array([3,2,1]), (0, 0, 0), 1)])
@pytest.mark.parametrize("x",
                         [(np.array([1, 4, .5])),
                          (np.array([6, 6, 120])),
                          (np.array([1000, 0, -40]))])
@pytest.mark.parametrize("func", [("pdf"),("cdf"),("logpdf"),("gradient")])                        
def test_InverseGamma(a, location, scale, x, func):
    IGD = cuqi.distribution.InverseGamma(a, location=location, scale=scale)

    # PDF formula for InverseGamma
    def my_pdf( x, a, location, scale):
        if np.any(x <= location):
            val = x*0
        else:
            val =  (scale**a)\
            /((x-location)**(a+1)*sp.special.gamma(a))\
            *np.exp(-scale/(x-location))
        return val

    if func == "pdf":
        # The joint PDF of independent random vairables is the product of their individual PDFs.
        print("#########")
        print(IGD.pdf(x))
        print(np.prod(my_pdf(x, IGD.shape, IGD.location, IGD.scale)))

        assert np.all(np.isclose(IGD.pdf(x),np.prod(sp.stats.invgamma.pdf(x, a=a, loc=location, scale=scale)))) and np.all(np.isclose(IGD.pdf(x),np.prod(my_pdf(x, IGD.shape, IGD.location, IGD.scale))))

    elif func == "cdf":
        # The joint CDF of independent random vairables is the product of their individual CDFs.
        assert np.all(np.isclose(IGD.cdf(x),np.prod(sp.stats.invgamma.cdf(x, a=a, loc=location, scale=scale))))

    elif func == "logpdf":
        # The joint PDF of independent random vairables is the product of their individual PDFs (the product is replaced by sum for logpdf).
        assert np.all(np.isclose(IGD.logpdf(x),np.sum(sp.stats.invgamma.logpdf(x, a=a, loc=location, scale=scale)))) and np.all(np.isclose(IGD.logpdf(x),np.sum(np.log(my_pdf(x, IGD.shape, IGD.location, IGD.scale)))))

    elif func == "gradient":
        FD_gradient = cuqi.utilities.approx_gradient(IGD.logpdf, x, epsilon=0.000000001)
        #Assert that the InverseGamma gradient is close to the FD approximation or both gradients are nan.
        assert (np.all(np.isclose(IGD.gradient(x),FD_gradient,rtol=1e-3)) or
          (np.all(np.isnan(FD_gradient)) and np.all(np.isnan(IGD.gradient(x)))))

    else:
        raise ValueError
    
@pytest.mark.parametrize("shape", [1, 2, 3, 5])
@pytest.mark.parametrize("rate", [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e4, 1e5])
@pytest.mark.parametrize("value", [1, 2, 3, 4, 5])
def test_Gamma_pdf(shape, rate, value):
    G = cuqi.distribution.Gamma(shape, rate)
    assert np.isclose(G.pdf(value), scipy_stats.gamma(shape, scale=1/rate).pdf(value))

@pytest.mark.parametrize("shape", [1, 2, 3, 5])
@pytest.mark.parametrize("rate", [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e4, 1e5])
@pytest.mark.parametrize("value", [1, 2, 3, 4, 5])
def test_Gamma_cdf(shape, rate, value):
    G = cuqi.distribution.Gamma(shape, rate)
    assert np.isclose(G.cdf(value), scipy_stats.gamma(shape, scale=1/rate).cdf(value))

@pytest.mark.parametrize("shape", [1, 2, 3, 5])
@pytest.mark.parametrize("rate", [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e4, 1e5])
def test_Gamma_sample(shape, rate):
    rng = np.random.RandomState(3)
    G = cuqi.distribution.Gamma(shape, rate)
    cuqi_samples = G.sample(3, rng=rng)

    rng2 = np.random.RandomState(3)
    np_samples = rng2.gamma(shape=shape, scale=1/rate, size=(3, 1)).T

    assert np.allclose(cuqi_samples.samples, np_samples)

@pytest.mark.parametrize("location", [-1, -2, -3, 0, 1, 2, 3])
@pytest.mark.parametrize("scale", [1e-3, 1e-1, 1e0, 1e1, 1e3])
@pytest.mark.parametrize("value", [1e-3, 1e-1, 1e0, 1e1, 1e3])
def test_Laplace_pdf(location, scale, value):
    LPL = cuqi.distribution.Laplace(location, scale)
    assert np.isclose(LPL.pdf(value), scipy_stats.laplace(location, scale).pdf(value))

@pytest.mark.xfail(reason="Expected to fail after fixing Gaussian sample. Regression needs to be updated")
def test_lognormal_sample():
    rng = np.random.RandomState(3)
    mean = np.array([0, -4])
    std = np.array([1, 14])
    R = np.array([[1, -0.7], [-0.7, 1]])
    LND = cuqi.distribution.Lognormal(mean, std**2*R)
    cuqi_samples = LND.sample(3,rng=rng)
    result = np.array([[5.92800307e+00, 1.54490711e+00, 1.09977286e+00],
                       [7.82563924e-14, 3.68757304e-04, 1.26980745e-04]])
    assert(np.all(np.isclose(cuqi_samples.samples, result)))

@pytest.mark.parametrize("mean,std",[
                        (np.array([0, 0]),
                        np.array([1, 1])),
                        (np.array([-3.14159265,  2.23606798]),
                        np.array([3.14159265, 50.  ])),
                        (np.array([1.   ,  0.001]),
                        np.array([1, 120]))
                        ])
@pytest.mark.parametrize("x",[
                        (np.array([100, 0.00001])),
                        (np.array([0, -.45])),
                        (np.array([-3.14159265, 2.23606798]))])
def test_lognormal_logpdf(mean,std, x ):

    # CUQI lognormal x1,x2
    R = np.array([[1, 0], [0, 1]])
    LND = cuqi.distribution.Lognormal(mean, std**2*R)
    
    # Scipy lognormal for x1
    x_1_pdf_scipy = sp.stats.lognorm.pdf(x[0], s = std[0], scale= np.exp(mean[0]))

    # Scipy lognormal for x2
    x_2_pdf_scipy = sp.stats.lognorm.pdf(x[1], s = std[1], scale= np.exp(mean[1]))
    
    # x1 and x2 are independent 
    assert(np.isclose(LND.pdf(x), x_1_pdf_scipy*x_2_pdf_scipy))
    assert(np.isclose(LND.logpdf(x), np.log(x_1_pdf_scipy*x_2_pdf_scipy)))

@pytest.mark.parametrize("mean, std, R",[
                        (np.array([0, 0]),
                        np.array([1, 1]),
                        np.array([[1, .3], [.3, 1]])),
                        (np.array([-3.14159265,  2.23606798]),
                        np.array([3.14159265, 50.  ]),
                        np.array([[1, 0], [0, 1]])),
                        (np.array([1.   ,  0.001]),
                        np.array([1, 120]),
                        np.array([[2, 2], [2, 14]]))
                        ])
@pytest.mark.parametrize("val",[
                        (np.array([100, 0.1])),
                        (np.array([-10, .1])),
                        (np.array([0.1, .45])),
                        (np.array([3.14159265, 2.23606798]))])
def test_gradient_lognormal_as_prior(mean, std, R, val):
    # This test verifies the lognormal distribution gradient correctness
    # (when the mean is a numpy.ndarray) by comparing it with a finite
    # difference approximation of the gradient.

    # ------------------- 1. Create lognormal distribution --------------------
    Sigma = np.diag(std)@R@np.diag(std) # std**2*R
    LND = cuqi.distribution.Lognormal(mean, Sigma)

    # -------------- 2. Create the finite difference gradient -----------------
    eps = 0.000001
    FD_gradient = np.empty(LND.dim)

    for i in range(LND.dim):
        # compute the ith component of the gradient
        eps_vec = np.zeros(LND.dim)
        eps_vec[i] = eps
        val_plus_eps = val + eps_vec
        FD_gradient[i] = (LND.logpdf(val_plus_eps) - LND.logpdf(val))/eps 

    # ---------------- 3. Verify correctness of the gradient ------------------
    assert(np.all(np.isclose(FD_gradient, LND.gradient(val))) or
           (np.all(np.isnan(FD_gradient)) and np.all(np.isnan(LND.gradient(val)))) )

@pytest.mark.parametrize("std, R",[
                        (
                        3,
                        np.array([[1, .3, 0], 
                                  [.3, 1, .3], 
                                  [0, .3, 1]])),
                        (
                        12,
                        np.array([[1, 0, 0], 
                                  [0, 1, 0], 
                                  [0, 0, 1]]))
                        ])
@pytest.mark.parametrize("val",[
                        (np.array([10, 0.1, 3])),
                        (np.array([0.1, .45, 6]))])
@pytest.mark.parametrize("x",[
                        (np.array([1, -0.1])),
                        (np.array([-3.14159265, 2.23606798]))])
                      
def test_gradient_lognormal_as_likelihood(std, R, val, x):
    # This test verifies the lognormal distribution gradient correctness
    # (when the mean is a cuqi.Model) by comparing it with a finite
    # difference approximation of the gradient.

    # ------------------------ 1. Create a cuqi.model -------------------------
    domain_geometry = 2
    range_geometry = 3

    # model's forward function
    def forward(x):
        return np.array([np.exp(x[0]) + x[1],
             np.exp(2*x[1]) + 3*x[0],
             x[0]+2*x[1]])
    
    # model's gradient
    def gradient(val, x=None):
        jac = np.zeros((range_geometry, domain_geometry))
        jac[0,0] =  np.exp(x[0])#d f1/ d x1
        jac[0,1] =  1 #d f1/ d x2
    
        jac[1,0] =  3 #d f2/ d x1
        jac[1,1] =  2*np.exp(2*x[1]) #d f2/ d x2
    
        jac[2,0] =  1 #d f2/ d x1
        jac[2,1] =  2 #d f2/ d x2
    
        return jac.T@val
    
    # create the cuqi.Model
    model = cuqi.model.Model(forward=forward, 
                             domain_geometry=domain_geometry, 
    			 range_geometry=range_geometry)
    model.gradient = gradient
    
    # ------------------- 2. Create lognormal distribution --------------------
    LND = cuqi.distribution.Lognormal(model, std**2*R)
    
    # -------------- 3. Create the finite difference gradient -----------------
    eps = 0.000001
    FD_gradient = np.empty(domain_geometry)
 
    for i in range(domain_geometry):
        eps_vec = np.zeros(domain_geometry)
        eps_vec[i] = eps
        x_plus_eps = x + eps_vec
        FD_gradient[i] = (LND(x=x_plus_eps).logpdf(val) - LND(x=x).logpdf(val))/eps
    
    # ---------------- 4. Verify correctness of the gradient ------------------
    assert(np.all(np.isclose(FD_gradient, LND.gradient(val, x=x))))

def test_beta(): #TODO. Make more tests
    alpha = 2; beta = 5; x=0.5

    # Create beta distribution
    BD = cuqi.distribution.Beta(alpha, beta)

    # Gamma function
    gamma = lambda x: sp.special.gamma(x)

    # Direct PDF formula for Beta
    def my_pdf( x, a, b):
        if np.any(x<=0) or np.any(x>=1) or np.any(a<=0) or np.any(b<=0):
            val = x*0
        else:
            val = gamma(a+b)/(gamma(a)*gamma(b)) * x**(a-1) * (1-x)**(b-1)
        return val

    # PDF
    assert np.allclose(BD.pdf(x), my_pdf(x, BD.alpha, BD.beta))

    # CDF
    assert np.allclose(BD.cdf(x),np.prod(sp.stats.beta.cdf(x, a=alpha, b=beta)))

    # logpdf
    assert np.allclose(BD.logpdf(x), np.log(my_pdf(x, alpha, beta)))
    
    # GRADIENT
    FD_gradient = cuqi.utilities.approx_gradient(BD.logpdf, x, epsilon=0.000000001)
    assert np.allclose(BD.gradient(x),FD_gradient,rtol=1e-3) or (np.all(np.isnan(FD_gradient)) and np.all(np.isnan(BD.gradient(x))))

# Fixture for beta distribution where beta is a likelihood
@pytest.fixture
def beta_likelihood():
    # simple forward model
    A = cuqi.model.Model(
        lambda x: x**2,
        range_geometry=1,
        domain_geometry=1,
        gradient=lambda direction, wrt: 2*wrt*direction)
    
    # set a gaussian prior
    x = cuqi.distribution.Gaussian(0, 1)
    # Beta data distribution
    y = cuqi.distribution.Beta(A(x),1)
    # set the observed data
    y=y(y=0.5)
    return y

def test_gradient_for_Beta_as_likelihood_raises_error(beta_likelihood):
    """Test computing the gradient of the Beta distribution as a likelihood
    raises a NotImplementedError"""

    with pytest.raises(NotImplementedError, 
                       match=r"Gradient is not implemented for CUQI Beta."):
        beta_likelihood.gradient(1)

@pytest.mark.parametrize("C",[1, np.ones(5), np.eye(5), sps.eye(5), sps.diags(np.ones(5))])
def test_Gaussian_Cov_sample(C):
    x = cuqi.distribution.Gaussian(np.zeros(5), np.pi*C)
    rng = np.random.RandomState(0)
    samples = x.sample(rng=rng)
    assert np.allclose(samples, np.array([3.12670137, 0.70926018, 1.73476791, 3.97187978, 3.31016035]))


@pytest.mark.parametrize("dist",
                         [cuqi.distribution.Gaussian(np.zeros(2), np.eye(2)),
                          cuqi.distribution.Beta(np.ones(2)*2, 5),
                          cuqi.distribution.Lognormal(np.ones(2)*.1, 4),
                          cuqi.distribution.Gaussian(np.zeros(2),
                                                     np.array([[1.0, 0.7],
                                                               [0.7,  1.]])),
                          cuqi.distribution.GMRF(np.zeros(2), 0.1)])
def test_enable_FD_gradient_distributions(dist):
    """Test that the distribution FD gradient is close to the exact gradient"""
    x = np.array([0.1, 0.3])

    # Exact gradient
    g_exact = dist.gradient(x)

    # FD gradient
    dist.enable_FD()
    g_FD = dist.gradient(x)

    # Assert that the FD gradient is close to the exact gradient
    # but not exactly equal to it
    assert np.allclose(g_exact, g_FD) and np.all(g_exact != g_FD)


@pytest.mark.parametrize("x",
                         [cuqi.distribution.Gaussian(np.zeros(6), np.eye(6)),
                          cuqi.distribution.Beta(np.ones(6)*2, 5),
                          cuqi.distribution.Lognormal(np.ones(6)*.1, 4),
                          cuqi.distribution.GMRF(np.zeros(6), 0.1)])
@pytest.mark.parametrize("y",
                         [cuqi.distribution.Gaussian(np.zeros(6), np.eye(6)),
                          cuqi.distribution.Lognormal(np.zeros(6), 4)])
@pytest.mark.parametrize("x_i", [np.array([0.1, 0.3, 6, 12, 1, 2]),
                                 np.array([0.1, 0.3, 0.5, 6, 3, 1])])
def test_enable_FD_gradient_posterior(x, y, x_i):
    """ Test that the posterior exact gradient and FD gradient are close."""

    # Create a model
    model = cuqi.testproblem.Deconvolution1D(dim=6).model

    # Create likelihood
    y.mean = model
    data = y(x_i).sample()
    likelihood = y(y=data)

    # Create posterior
    posterior = cuqi.distribution.Posterior(likelihood, x)

    # Compute exact gradient
    g_exact = posterior.gradient(x_i)

    # Compute FD gradient
    posterior.enable_FD(1e-7)
    g_FD = posterior.gradient(x_i)

    # Assert that the exact and FD gradient are close,
    # but not exactly equal (since we use a different method)
    assert np.allclose(g_exact, g_FD) and np.all(g_exact != g_FD)\
        or (np.all(np.isnan(g_exact)) and np.all(np.isnan(g_FD)))

def test_Cauchy_against_scipy():
    """ Test Cauchy distribution logpdf, cdf, pdf, gradient """

    x = cuqi.distribution.Cauchy(np.random.randn(5), np.abs(np.random.rand(5)))

    val = np.random.randn(5)

    # Test logpdf
    assert np.allclose(x.logpdf(val), np.sum(scipy_stats.cauchy.logpdf(val, loc=x.location, scale=x.scale)))

    # Test pdf
    assert np.allclose(x.pdf(val), np.prod(scipy_stats.cauchy.pdf(val, loc=x.location, scale=x.scale)))

    # Test cdf
    assert np.allclose(x.cdf(val), np.sum(scipy_stats.cauchy.cdf(val, loc=x.location, scale=x.scale)))

    # Test gradient
    assert np.allclose(x.gradient(val), cuqi.utilities.approx_derivative(x.logpdf, val))

def test_Cauchy_out_of_range_values():
    """ Test that the logpdf is -inf for values outside the support """

    x = cuqi.distribution.Cauchy(np.random.randn(5), np.abs(np.random.rand(5)))

    x.scale[0] = 0 # This is not a valid scale

    val = np.random.randn(5)

    # Test logpdf
    assert np.allclose(x.logpdf(val), -np.inf)

def test_Gaussian_sqrtprec_must_be_square():
    """ Test Gaussian distribution raises ValueError if sqrtprec (sparse and dense) is not square """
    N = 10; M = 5

    # Create non square sqrtprec as sparse (later converted to dense)
    sqrtprec = sp.sparse.csr_matrix(np.random.randn(N, M))

    with pytest.raises(ValueError, match="sqrtprec must be square"):
        cuqi.distribution.Gaussian(mean = np.zeros(N), sqrtprec = sqrtprec)

    with pytest.raises(ValueError, match="sqrtprec must be square"):
        cuqi.distribution.Gaussian(mean = np.zeros(N), sqrtprec = sqrtprec.todense())

def test_Gaussian_from_sparse_sqrtprec():
    """ Test Gaussian distribution from sparse sqrtprec is equal to dense sqrtprec """
    N = 10; M = 5

    sqrtprec = sp.sparse.spdiags(np.random.randn(N), 0, N, N)

    y_from_sparse = cuqi.distribution.Gaussian(mean = np.zeros(N), sqrtprec = sqrtprec)
    y_from_dense = cuqi.distribution.Gaussian(mean = np.zeros(N), sqrtprec = sqrtprec.todense())

    assert y_from_dense.logpdf(np.ones(N)) == y_from_sparse.logpdf(np.ones(N))

def test_Gaussian_from_linear_operator_sqrtprec():
    """ Test Gaussian distribution from LinearOperator sqrtprec is equal to dense sqrtprec """
    N = 10; M = 5

    sqrtprec = sp.sparse.spdiags(np.random.randn(N), 0, N, N)
    prec = sqrtprec.todense()@sqrtprec.todense().T

    def matvec(x):
        return sqrtprec @ x
    def rmatvec(x):
        return sqrtprec.T @ x
    
    sqrtprec_operator = sp.sparse.linalg.LinearOperator((N, N), matvec=matvec, rmatvec=rmatvec)
    sqrtprec_operator.logdet = -np.log(nplinalg.det(prec))

    y_from_sparse = cuqi.distribution.Gaussian(mean = np.zeros(N), sqrtprec = sqrtprec_operator)
    y_from_dense = cuqi.distribution.Gaussian(mean = np.zeros(N), sqrtprec = sqrtprec.todense())

    assert np.allclose(y_from_dense.logpdf(np.ones(N)), y_from_sparse.logpdf(np.ones(N)))

@pytest.mark.parametrize("alpha, beta, gamma, expected",[
                        (1.0, 2.0, 3.0, [[0.77974597], [0.77361298], [0.5422682 ], [0.81054637], [1.35205349]]),
                        (128.0, 3.0, -4.0, [[4.80290413], [4.40134124], [4.56151431], [4.2430851 ], [4.50618522]])
                        ])
def test_MHN_sample_regression(alpha, beta, gamma, expected):
    rng = np.random.RandomState(0)
    dist = cuqi.distribution.ModifiedHalfNormal(alpha, beta, gamma)
    samples = dist.sample(5,rng=rng).samples
    assert np.allclose( samples, np.array(expected))

@pytest.mark.parametrize("alpha, beta, gamma, expected_logpdf, expected_gradient",[
                        (1.0, 2.0, 3.0, -65.0, [[-1.0], [-5.0], [-9.0], [-13.0], [-17.0]]),
                        (64.0, 3.0, -4.0, 76.6119797952689, [[53.0], [15.5], [-1.0], [-12.25], [-21.4]])
                        ])
def test_MHN_regression(alpha, beta, gamma, expected_logpdf, expected_gradient):
    dist = cuqi.distribution.ModifiedHalfNormal(alpha, beta, gamma)
    logpdf = dist.logpdf(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    gradient = dist._gradient(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert np.allclose( logpdf, np.array(expected_logpdf))
    assert np.allclose( gradient, np.array(expected_gradient))

def test_Smoothed_Laplace():
    """ Test Smoothed Laplace distribution logpdf and gradient """

    location = np.array([1, 2])
    scale = np.array([1, 2])
    scalar_laplace_0 = cuqi.distribution.Laplace(location[0], scale[0])

    scalar_smoothed_laplace_0 = cuqi.distribution.SmoothedLaplace(location[0], scale[0], 1e-8)
    scalar_smoothed_laplace_1 = cuqi.distribution.SmoothedLaplace(location[1], scale[1], 1e-8)
    vector_smoothed_laplace = cuqi.distribution.SmoothedLaplace(location, scale, 1e-8)

    x = np.array([3, 4])

    # logpdf (scalar Laplace vs scalar Smoothed Laplace)
    assert np.allclose(scalar_laplace_0.logpdf([x[0]]), scalar_smoothed_laplace_0.logpdf(x[0]))

    # logpdf (scalar Smoothed Laplace * scalar Smoothed Laplace vs vector Smoothed Laplace)
    assert np.allclose(scalar_smoothed_laplace_0.logpdf(x[0])+scalar_smoothed_laplace_1.logpdf(x[1]),
                       vector_smoothed_laplace.logpdf(x))
    
    # gradient (scalar Smoothed Laplace vs analytical)
    assert np.allclose(scalar_smoothed_laplace_0.gradient(x[0]), -1/scale[0])

    # gradient (vector Smoothed Laplace vs analytical)
    assert np.allclose(vector_smoothed_laplace.gradient(x), -1/scale)
