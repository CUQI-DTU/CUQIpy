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
