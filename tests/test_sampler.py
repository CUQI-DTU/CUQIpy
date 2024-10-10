import cuqi
import numpy as np

import sys

from cuqi.distribution import Gaussian, CMRF, Gaussian, LMRF, GMRF
from cuqi.sampler import pCN

import pytest


def test_CWMH_modify_proposal():
    # Parameters
    n = 2
    scale = 0.05*np.ones(n)
    x0 = 0.5*np.ones(n)

    # Set up target
    target_dist = cuqi.distribution.CMRF(np.zeros(n), 0.5, 'neumann')
    def target(x): return target_dist.pdf(x)

    # Set up proposals
    #proposal1 = cuqi.distribution.Normal(np.zeros(n),1 )
    #proposal2 = cuqi.distribution.Normal(np.zeros(n),2 )
    #def proposal1(x, sigma): return np.random.normal(x, sigma)
    #def proposal2(x, sigma): return np.random.normal(x, 2*sigma)

    proposal1 =cuqi.distribution.Normal(mean = lambda location:location,std = lambda scale:scale, geometry=n)
    proposal2 =cuqi.distribution.Normal(mean = lambda location:location,std = lambda scale:2*scale, geometry=n)


    # Set up sampler
    MCMC1 = cuqi.sampler.CWMH(target, proposal1, scale, x0)
    MCMC2 = cuqi.sampler.CWMH(target, proposal2, scale, x0)

    # Switch proposal
    MCMC1.proposal = proposal2

    # Compute samples 1
    np.random.seed(0)
    results1 = MCMC1.sample(10,2).samples

    # Compute samples 2
    np.random.seed(0)
    results2 = MCMC2.sample(10,2).samples

    # Compare samples
    assert np.allclose(results1,results2)

def test_CWMH_sample_regression():
    # Set seed
    np.random.seed(0)
    
    # Define target
    mean = np.array([0, 0])
    std = np.array([1, 1])
    R = np.array([[1, -0.7], [-0.7, 1]])
    cov = np.diag(std) @ (R @ np.diag(std))
    def target(x): return cuqi.distribution.Gaussian(mean,cov).pdf(x)

    # Define proposal
    # def proposal(x, sigma): return np.random.normal(x, sigma)
    #proposal = cuqi.distribution.Normal(np.zeros(len(mean)),1 )
    proposal =cuqi.distribution.Normal(mean = lambda location:location,std = lambda scale:scale, geometry=2)

    # Set up sampler
    MCMC = cuqi.sampler.CWMH(target, proposal, 0.05, np.array([0,0]))

    # Compare with previously computed results
    np.allclose(MCMC.sample(5,1).samples,np.array([[ 0.18158052,  0.17641957,  0.21447146,  0.23666462,  0.23666462],[-0.02885603, -0.00832611, -0.00224236,  0.01444136,  0.01444136]]))


def test_RWMH_sample_regression():
    # Set seed
    np.random.seed(0) #nice seed!

    d = 6
    mu = np.zeros(d)
    sigma = np.linspace(0.5, 1, d)

    # target function to sample
    dist = cuqi.distribution.Gaussian(mu, sigma**2)

    ref = cuqi.distribution.Gaussian(mu, np.ones(d))   # standard Gaussian

    scale = 0.1
    x0 = 0.5*np.ones(d)
    MCMC2 = cuqi.sampler.MH( dist,proposal = ref,scale =scale, x0=x0)

    # run sampler
    Ns = int(1e1)      # number of samples
    Nb = int(0.2*Ns)   # burn-in

    #
    x_s2 = MCMC2.sample_adapt(Ns, Nb)
    reference = np.array([[ 0.5       ,  0.77238519,  0.73381779,  0.7700134 ,  0.41274389,
         0.18348216,  0.37057737,  0.34837564,  0.34837564,  0.34837564],
       [ 0.5       ,  0.46259765,  0.70379628,  0.79213478,  1.00263215,
         0.80556899,  0.33926608,  0.67670237,  0.67670237,  0.67670237],
       [ 0.5       ,  0.5570753 ,  0.50402891,  0.29664278,  0.16967913,
        -0.39387781, -0.57857024, -0.52616853, -0.52616853, -0.52616853],
       [ 0.5       ,  0.34429001,  0.35183386, -0.1108788 , -0.366134  ,
        -0.01638806, -0.23662861,  0.00165624,  0.00165624,  0.00165624],
       [ 0.5       ,  0.03456505,  0.2675372 ,  0.18626517,  0.51389191,
        -0.21210323,  0.46072045, -0.03899323, -0.03899323, -0.03899323],
       [ 0.5       ,  0.61916105,  0.86364669,  0.9001697 ,  0.26407212,
         0.16837312, -0.10678787, -0.39255235, -0.39255235, -0.39255235]])

    assert np.allclose(x_s2.samples,reference)

def test_pCN_sample_regression():
    np.random.seed(0)
    d= 2
    mu = np.zeros(d)
    sigma = np.linspace(0.5, 1, d)
    model = cuqi.model.Model(lambda x: x, range_geometry=d, domain_geometry=d)
    L = Gaussian(mean=model, sqrtcov=sigma).to_likelihood(np.zeros(d))
    def target(x): return L.logd(x)
    P = Gaussian(mu, np.ones(d))
    scale = 0.1
    x0 = 0.5*np.ones(d)
    posterior = cuqi.distribution.Posterior(L, P)
    MCMC = pCN(posterior, scale, x0)
    results = MCMC.sample(10,2)
    assert np.allclose(results.samples, np.array([[0.44368817, 0.44368817, 0.56807601, 0.64133227, 0.64133227,
         0.78752546, 0.68750247, 0.42875734, 0.40239322, 0.40495205],
        [0.66816968, 0.66816968, 0.61423278, 0.6233214 , 0.6233214 ,
         0.59968114, 0.6343679 , 0.69654995, 0.84449757, 0.82154609]]))

    assert np.allclose(results.loglike_eval, np.array([-1.76167363, -1.76167363, -1.97879155, -2.16160882, -2.16160882,
        -2.5649313 , -2.2912605 , -1.75498652, -1.82515857, -1.8101712 ]))

    assert results.acc_rate==0.8


def test_sampler_geometry_assignment():

    n = 2
    scale = 0.05*np.ones(n)
    x0 = 0.5*np.ones(n)

    # Set up target
    target = cuqi.distribution.CMRF(np.zeros(n), 0.5, 'neumann')
    target.geometry = cuqi.geometry.Continuous2D((1,2))

    # Set up proposals
    proposal =cuqi.distribution.Normal(mean = lambda location:location,std = lambda scale:scale, geometry=n)

    # Set up sampler
    MCMC_sampler = cuqi.sampler.CWMH(target, proposal, scale, x0)
    samples = MCMC_sampler.sample(10,2)

    assert(MCMC_sampler.geometry == target.geometry and\
           MCMC_sampler.geometry == samples.geometry)

def test_sampler_UserDefined_basic():
    """
    This tests all samplers which can work with any logpdf as input 
    (potentially requireing a gradient or sample method also)
    """
    # Distribution
    X = cuqi.distribution.Gaussian(np.array([2,3]),np.array([[2,0.1],[0.1,5]]))
    distX = cuqi.distribution.UserDefinedDistribution(X.dim, X.logpdf, X.gradient, X.sample)

    # Parameters
    Ns = 500   # number of samples
    Nb = 100   # burn-in

    s_MH = cuqi.sampler.MH(distX).sample_adapt(Ns,Nb)
    s_CWMH = cuqi.sampler.CWMH(distX).sample_adapt(Ns,Nb)
    s_NUTS = cuqi.sampler.NUTS(distX).sample_adapt(Ns,Nb)

    assert np.allclose(s_MH.shape,(X.dim,Ns))
    assert np.allclose(s_CWMH.shape,(X.dim,Ns))
    assert np.allclose(s_NUTS.shape,(X.dim,Ns))

def test_sampler_UserDefined_tuple():
    """
    This tests samplers with a two userdefined distributions.
    Here we require two userdefined distributions.
    """
    # This provides a way to give the logpdf
    P = cuqi.distribution.Gaussian(np.array([2,3]),np.array([[2,0.1],[0.1,5]]))

    #
    model = cuqi.model.Model(lambda x: x, range_geometry=2, domain_geometry=2)
    L = cuqi.distribution.Gaussian(model,np.array([[1,0.5],[0.5,3]])).to_likelihood(np.array([5,6]))

    # Define userdefined distribution + likelihood
    userP = cuqi.distribution.UserDefinedDistribution(2, P.logpdf, P.gradient, P.sample)

    userL = cuqi.likelihood.UserDefinedLikelihood(2, L.logd, L.gradient)

    # Parameters
    Ns = 2000   # number of samples
    Nb = 200   # burn-in

    # Run samplers
    s_pCN = cuqi.sampler.pCN((userL,userP)).sample_adapt(Ns)

    assert np.allclose(s_pCN.shape,(P.dim,Ns))

def test_sampler_CustomInput_LinearRTO():
    """
    This tests the LinearRTO sampler which requires a specific input structure.
    """
    # In some special cases A model must be part of the likelihood (e.g. in LinearRTO)
    model = np.eye(2) # Identity model
    #model = cuqi.model.LinearModel(lambda x: x, lambda y: y, 2, 2) 

    # In LinearRTO we require Gaussian distributions
    # or at least classes with sqrtprec and sqrtprecTimesMean 
    # This is most easily done by defining a Gaussian. #TODO make UserDefined.
    P = cuqi.distribution.Gaussian(np.array([2,3]),np.array([[2,0.1],[0.1,5]]))
    L = cuqi.distribution.Gaussian(model,np.array([[1,0.5],[0.5,3]]))

    # Data
    data = np.array([5,6])

    # Posterior
    target = (data, model, L.sqrtprec, P.mean, P.sqrtprec)
    #target = cuqi.distribution.Posterior(L, P, data)

    # Parameters
    Ns = 2000   # number of samples
    Nb = 200   # burn-in

    # Sampling
    s_RTO = cuqi.sampler.LinearRTO(target).sample_adapt(Ns,Nb)

    assert np.allclose(s_RTO.shape,(P.dim,Ns))

def test_sampler_scalar_mean_Gaussian_LinearRTO():

    model = np.eye(2) # Identity model

    P = cuqi.distribution.Gaussian(0, 1, geometry=2)
    L = cuqi.distribution.Gaussian(model,np.array([[1,0.5],[0.5,3]]))

    # Data
    data = np.array([5,6])

    # Posterior
    target = (data, model, L.sqrtprec, P.mean, P.sqrtprec)

    # Parameters
    Ns = 200   # number of samples
    Nb = 20   # burn-in

    # Sampling
    s_RTO = cuqi.sampler.LinearRTO(target).sample_adapt(Ns,Nb)

    assert np.allclose(s_RTO.shape,(P.dim,Ns))


def test_ULA_UserDefinedDistribution():
    expected_samples = \
        np.array([[0.1, 0.11763052, 0.12740614],
                  [1.1, 1.10399157, 1.1263901 ]])
    np.random.seed(0)
    # Parameters
    dim = 2 # Dimension of distribution
    mu = np.arange(dim) # Mean of Gaussian
    std = 1 # standard deviation of Gaussian

    # Logpdf function
    logpdf_func = lambda x: -1/(std**2)*np.sum((x-mu)**2)
    gradient_func = lambda x: -2/(std**2)*(x - mu)

    # Define distribution from logpdf as UserDefinedDistribution (sample and gradients also supported)
    target = cuqi.distribution.UserDefinedDistribution(dim=dim, logpdf_func=logpdf_func, gradient_func=gradient_func)

    # Set up sampler
    sampler = cuqi.sampler.ULA(target, scale=.0001, x0=np.array([.1, 1.1]))

    # Sample
    samples = sampler.sample(3)

    assert np.allclose(samples.samples, expected_samples) and np.isclose(samples.acc_rate, 1)


def test_ULA_regression(copy_reference):
    # This tests compares ULA class results with results 
    # generate from original ULA code provided by Felipe Uribe.
    # The original code is found in commit:
    # c172442d8d7f34a33681b9c1d76889c99ac8dfcd

    np.random.seed(0)
    # %% Create CUQI test problem
    test = cuqi.testproblem._Deblur()
    n = test.model.domain_dim
    h = test.meshsize
    
    # Extract data
    data = test.data
    
    # Extract Likelihood
    likelihood  = test.likelihood
    
    # Define Prior
    loc = np.zeros(n)
    delta = 1
    scale = delta*h
    prior = cuqi.distribution.CMRF(loc, scale, 'neumann')
    
    # %% Create the posterior and the sampler
    posterior = cuqi.distribution.Posterior(likelihood, prior)
    MCMC = cuqi.sampler.ULA(posterior, scale=0.0001)

    # %% Sample
    samples  = MCMC.sample(5)
    samples_orig_file = copy_reference("data/ULA_felipe_original_code_results.npz")
    samples_orig = np.load(samples_orig_file)

    assert(np.allclose(samples.samples, samples_orig['arr_0']))


def test_MALA_UserDefinedDistribution():
    expected_samples = \
        np.array([[0.1, 0.11763052, 0.09493548],
                  [1.1, 1.10399157, 1.11731663]])
    np.random.seed(0)

    # Parameters
    dim = 2  # Dimension of distribution
    mu = np.arange(dim)  # Mean of Gaussian
    std = 1  # standard deviation of Gaussian

    # Logpdf function
    logpdf_func = lambda x: -1/(std**2)*np.sum((x-mu)**2)
    gradient_func = lambda x: -2/(std**2)*(x - mu)

    # Define distribution from logpdf as UserDefinedDistribution (sample and gradients also supported)
    target = cuqi.distribution.UserDefinedDistribution(
        dim=dim, logpdf_func=logpdf_func, gradient_func=gradient_func)

    # Set up sampler
    sampler = cuqi.sampler.MALA(target, scale=.0001, x0=np.array([.1, 1.1]))

    # Sample
    samples = sampler.sample(3)

    assert np.allclose(samples.samples, expected_samples)\
                    and np.isclose(samples.acc_rate, 1)


def test_MALA_regression(copy_reference):
    #%% CUQI
    dim = 5  # Dimension of distribution
    mu = np.arange(dim)  # Mean of Gaussian
    std = 1  # standard deviation of Gaussian
    N = 2000
    Nb = 500
    eps = 1/dim

    # Logpdf function
    logpdf_func = lambda x: -1/(std**2)*np.sum((x-mu)**2)
    gradient_func = lambda x: -2/(std**2)*(x-mu)
    
    # Define distribution from logpdf as UserDefinedDistribution (sample and gradients also supported)
    target = cuqi.distribution.UserDefinedDistribution(dim=dim, logpdf_func=logpdf_func,
                                                       gradient_func=gradient_func)

    # Set up sampler
    x0 = np.zeros(dim)
    np.random.seed(0)
    sampler = cuqi.sampler.MALA(target, scale=eps**2, x0=x0)
    # Sample
    samples = sampler.sample(N, Nb)

    samples_orig_file = copy_reference(
        "data/MALA_felipe_original_code_results.npz")
    samples_orig = np.load(samples_orig_file)

    assert(np.allclose(samples.samples, samples_orig['arr_0']))

@pytest.mark.parametrize("prior, sample_method, expected", [
    (Gaussian(np.zeros(128), 0.1), "_sampleMapCholesky", np.arange(10)), # Direct (no burn-in, no initial guess)
    (Gaussian(np.zeros(128), 0.1), "_sampleLinearRTO", np.arange(1,12)), # 20% burn-in + initial guess
    (Gaussian(np.zeros(128), 0.1), "_sampleNUTS", np.arange(1,12)),      # 20% burn-in + initial guess
    (Gaussian(np.zeros(128), 0.1), "_samplepCN", np.arange(1,12)),       # 20% burn-in + initial guess
    (Gaussian(np.zeros(128), 0.1), "_sampleCWMH", np.arange(1,12)),      # 20% burn-in + initial guess
    (LMRF(0, 0.1, geometry=128),"_sampleUGLA", np.arange(1,12)),   # 20% burn-in + initial guess
    ])
def test_TP_callback(prior, sample_method, expected):
    """ Test that the callback function is called with the correct sample index by comparing to the expected output.
    
    This tests the pipeline from testproblem all the way to the sampler.
    """

    TP = cuqi.testproblem.Deconvolution1D(dim=prior.dim)
    TP.prior = prior

    # Define callback that stores a list of sample indices
    Ns_list = []
    def callback(_, n):
        Ns_list.append(n)

    # Sampler
    sample_method_handle = getattr(TP, sample_method)
    
    # if sampler is MapCholesky, we do not pass burn-in
    # otherwise we pass burn-in Nb
    if sample_method == "_sampleMapCholesky":
        sample_method_handle(10, callback=callback)
    else:
        sample_method_handle(10, Nb=2, callback=callback)

    assert np.array_equal(Ns_list, expected)

def test_NUTS_regression(copy_reference):
    # SKIP NUTS test if not windows (for now) #TODO.
    if  not sys.platform.startswith('win') and \
       not sys.platform.startswith('darwin'):
        pytest.skip("NUTS regression test is not implemented for this platform")

    np.random.seed(0)
    tp = cuqi.testproblem.WangCubic()
    x0 = np.ones(tp.model.domain_dim)
    Ns = int(1e3)
    Nb = int(0.5*Ns)
    MCMC = cuqi.sampler.NUTS(tp.posterior, x0, max_depth = 12)
    samples = MCMC.sample(Ns, Nb)

    if sys.platform.startswith('win'):
        samples_orig_file = copy_reference(
            "data/NUTS_felipe_original_code_results_win.npz")
    elif sys.platform.startswith('darwin'):
        samples_orig_file = copy_reference(
            "data/NUTS_felipe_original_code_results_darwin.npz")
    
    samples_orig = np.load(samples_orig_file)

    assert(np.allclose(samples.samples, samples_orig["arr_0"]))

def _Gibbs_joint_hier_model(use_legacy=False, noise_std=0.01):
    """ Define a Gibbs sampler based on a joint distribution from a hierarchical model. Used for testing Gibbs sampler. """
    np.random.seed(0)
    
    # Model and data
    A, y_obs, _ = cuqi.testproblem.Deconvolution1D(phantom='square', use_legacy=use_legacy, noise_std=noise_std).get_components()
    n = A.domain_dim

    # Define distributions
    d = cuqi.distribution.Gamma(1, 1e-4)
    l = cuqi.distribution.Gamma(1, 1e-4)
    x = cuqi.distribution.GMRF(np.zeros(n), lambda d: d[0])
    y = cuqi.distribution.Gaussian(A, lambda l: 1/l)

    # Combine into a joint distribution and create posterior
    joint = cuqi.distribution.JointDistribution(d, l, x, y)
    posterior = joint(y=y_obs)

    # Define sampling strategy
    sampling_strategy = {
        'x': cuqi.sampler.LinearRTO,
        ('d', 'l'): cuqi.sampler.Conjugate,
    }

    # Define Gibbs sampler
    sampler = cuqi.sampler.Gibbs(posterior, sampling_strategy)

    return sampler

def test_Gibbs_regression(copy_reference):

    # SKIP Gibbs reg test if not windows.
    if not sys.platform.startswith('win'):
        pytest.skip("NUTS regression test is not implemented for this platform")

    # Legacy deconvolution test problem to match reference results
    sampler = _Gibbs_joint_hier_model(use_legacy=True, noise_std=0.05)

    # Run sampler
    samples = sampler.sample(Ns=100, Nb=20)

    if sys.platform.startswith('win'):
        samples_orig_file = copy_reference(
            "data/Gibbs_original_code_results_win.npz")
    samples_orig = np.load(samples_orig_file)

    # Save results
    #np.savez("Gibbs_original_code_results_win.npz", samples["d"].samples)

    assert(np.allclose(samples["d"].samples, samples_orig["arr_0"]))

def test_Gibbs_continue_sampling():
    """ This tests the sampling can continue with the Gibbs sampler """

    sampler = _Gibbs_joint_hier_model()

    # Run sampler
    samples = sampler.sample(Ns=10, Nb=5)

    # Continue sampling
    samples2 = sampler.sample(Ns=10)

    assert samples["x"].shape[-1] == 10
    assert samples2["x"].shape[-1] == 20

def test_Gibbs_geometry_matches():
    sampler = _Gibbs_joint_hier_model()

    target = sampler.target

    # Run sampler
    samples = sampler.sample(Ns=10, Nb=5)

    # Check that the geometry matches
    assert samples["d"].geometry == target.get_density("d").geometry
    assert samples["l"].geometry == target.get_density("l").geometry
    assert samples["x"].geometry == target.get_density("x").geometry

def test_RTO_with_AffineModel_is_equivalent_to_LinearModel_and_shifted_data():
    # Define LinearModel and data
    A, y_obs, _ = cuqi.testproblem.Deconvolution1D().get_components()

    # Define Shift
    shift = np.random.rand(A.domain_dim)

    # Define Bayesian Problem
    x = cuqi.distribution.GMRF(np.zeros(A.domain_dim), 100)
    y = cuqi.distribution.Gaussian(A@x, 0.01**2)
    posterior = cuqi.distribution.JointDistribution(x, y)(y=y_obs-shift)

    # Set up LinearRTO with both models
    sampler_linear = cuqi.sampler.LinearRTO(posterior)

    # Sample with fixes seed
    np.random.seed(0)
    samples_linear = sampler_linear.sample(10, 2)

    # Define AffineModel
    affine_model = cuqi.model.AffineModel(A, shift)

    # Set up LinearRTO with AffineModel
    y = cuqi.distribution.Gaussian(affine_model, 0.01**2)
    posterior_affine = cuqi.distribution.JointDistribution(x, y)(y=y_obs)

    # Set up LinearRTO with AffineModel
    sampler_affine = cuqi.sampler.LinearRTO(posterior_affine)

    # Sample with fixes seed
    np.random.seed(0)
    samples_affine = sampler_affine.sample(10, 2)

    # Check that the samples are the same
    assert np.allclose(samples_linear.samples, samples_affine.samples)

def test_RegularizedRTO_with_AffieModel_is_equivalent_to_LinearModel_and_shifted_data():
    """ Test that sampling with RegularizedRTO with an AffineModel is equivalent to sampling with LinearModel and shifted data. """

    # Define LinearModel and data
    A, y_obs, _ = cuqi.testproblem.Deconvolution1D().get_components()

    # Define Shift
    shift = np.random.rand(A.domain_dim)

    # Define Bayesian Problem
    x = cuqi.implicitprior.NonnegativeGMRF(np.zeros(A.domain_dim), 100)
    y = cuqi.distribution.Gaussian(A@x, 0.01**2)
    posterior = cuqi.distribution.JointDistribution(x, y)(y=y_obs-shift)

    # Set up RegularizedRTO with both models
    sampler_linear = cuqi.sampler.RegularizedLinearRTO(posterior)

    # Sample with fixes seed
    np.random.seed(0)
    samples_linear = sampler_linear.sample(10, 2)

    # Define AffineModel
    affine_model = cuqi.model.AffineModel(A, shift)

    # Set up RegularizedRTO with AffineModel
    y = cuqi.distribution.Gaussian(affine_model, 0.01**2)
    posterior_affine = cuqi.distribution.JointDistribution(x, y)(y=y_obs)

    # Set up RegularizedRTO with AffineModel
    sampler_affine = cuqi.sampler.RegularizedLinearRTO(posterior_affine)

    # Sample with fixes seed
    np.random.seed(0)
    samples_affine = sampler_affine.sample(10, 2)

    # Check that the samples are the same
    assert np.allclose(samples_linear.samples, samples_affine.samples)
        
def test_UGLA_with_AffineModel_is_equivalent_to_LinearModel_and_shifted_data():
    """ Test that sampling with UGLA with an AffineModel is equivalent to sampling with LinearModel and shifted data. """

    # Define LinearModel and data
    A, y_obs, _ = cuqi.testproblem.Deconvolution1D().get_components()

    # Define Shift
    shift = np.random.rand(A.domain_dim)

    # Define Bayesian Problem
    x = cuqi.distribution.LMRF(np.zeros(A.domain_dim), 0.01)
    y = cuqi.distribution.Gaussian(A@x, 0.01**2)
    posterior = cuqi.distribution.JointDistribution(x, y)(y=y_obs-shift)

    # Set up UGLA with both models
    sampler_linear = cuqi.sampler.UGLA(posterior)

    # Sample with fixes seed
    np.random.seed(0)
    samples_linear = sampler_linear.sample(10, 2)

    # Define AffineModel
    affine_model = cuqi.model.AffineModel(A, shift)

    # Set up UGLA with AffineModel
    y = cuqi.distribution.Gaussian(affine_model, 0.01**2)
    posterior_affine = cuqi.distribution.JointDistribution(x, y)(y=y_obs)

    # Set up UGLA with AffineModel
    sampler_affine = cuqi.sampler.UGLA(posterior_affine)

    # Sample with fixes seed
    np.random.seed(0)
    samples_affine = sampler_affine.sample(10, 2)

    # Check that the samples are the same
    assert np.allclose(samples_linear.samples, samples_affine.samples)
