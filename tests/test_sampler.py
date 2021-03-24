import cuqi
import numpy as np

from pytest import approx

def test_CWMH_modify_proposal():
    # Parameters
    n = 2
    scale = 0.05*np.ones(n)
    x0 = 0.5*np.ones(n)

    # Set up target
    target_dist = cuqi.distribution.Cauchy_diff(np.zeros(n), 0.5, 'neumann')
    def target(x): return target_dist.pdf(x)

    # Set up proposals
    def proposal1(x, sigma): return np.random.normal(x, sigma)
    def proposal2(x, sigma): return np.random.normal(x, 2*sigma)

    # Set up sampler
    MCMC1 = cuqi.sampler.CWMH(target, proposal1, scale, x0)
    MCMC2 = cuqi.sampler.CWMH(target, proposal2, scale, x0)

    # Switch proposal
    MCMC1.proposal = proposal2

    # Compute samples 1
    np.random.seed(0)
    results1 = MCMC1.sample(10,2)

    # Compute samples 2
    np.random.seed(0)
    results2 = MCMC2.sample(10,2)

    # Compare samples
    assert np.allclose(results1[0],results2[0])

def test_CWMH_sample_regression():
    # Set seed
    np.random.seed(0)
    
    # Define target
    mean = np.array([0, 0])
    std = np.array([1, 1])
    R = np.array([[1, -0.7], [-0.7, 1]])
    def target(x): return cuqi.distribution.Gaussian(mean,std,R).pdf(x)

    # Define proposal
    def proposal(x, sigma): return np.random.normal(x, sigma)

    # Set up sampler
    MCMC = cuqi.sampler.CWMH(target, proposal, 0.05, np.array([0,0]))

    # Compare with previously computed results
    np.allclose(MCMC.sample(5,1)[0],np.array([[ 0.18158052,  0.17641957,  0.21447146,  0.23666462,  0.23666462],[-0.02885603, -0.00832611, -0.00224236,  0.01444136,  0.01444136]]))

def test_RWMH_sample_regression():
    # Set seed
    np.random.seed(0) #nice seed!

    d = 6
    mu = np.zeros(d)
    sigma = np.linspace(0.5, 1, d)
    R = np.eye(d)

    # target function to sample
    dist = cuqi.distribution.Gaussian(mu, sigma, R)
    def target(x): return dist.logpdf(x)

    ref = cuqi.distribution.Gaussian(mu, np.ones(d), R)   # standard Gaussian

    scale = 0.1
    x0 = 0.5*np.ones(d)
    MCMC2 = cuqi.sampler.RWMH(ref, target, scale, x0)

    # run sampler
    Ns = int(1e1)      # number of samples
    Nb = int(0.2*Ns)   # burn-in

    #
    x_s2, target_eval2, acc2 = MCMC2.sample_adapt(Ns, Nb)
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

    assert np.allclose(x_s2,reference)
