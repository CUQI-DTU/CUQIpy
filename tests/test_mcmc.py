import cuqi
import pytest
import numpy as np

def assert_true_if_sampling_is_equivalent(sampler_old: cuqi.sampler.Sampler, sampler_new: cuqi.mcmc.SamplerNew, Ns=100, atol=1e-1):
    """ Assert that the samples from the old and new sampler are equivalent. """

    np.random.seed(0)
    samples_old = sampler_old.sample(Ns).samples

    np.random.seed(0)
    samples_new = sampler_new.sample(Ns).get_samples().samples[...,:-1]

    assert np.allclose(samples_old, samples_new, atol=atol), f"Old: {samples_old}\nNew: {samples_new}"

def assert_true_if_warmup_is_equivalent(sampler_old: cuqi.sampler.Sampler, sampler_new: cuqi.mcmc.SamplerNew, Ns=100, Nb=100, Na=10, atol=1e-1):
    """ Assert that the samples from the old and new sampler are equivalent.
     
    Ns: int
        Number of samples.
    
    Nb: int
        Number of burn-in samples. (to be removed from the samples)

    Na: int
        Number of iterations to adapt.
              
    """

    # Get Ns samples from the old sampler
    # Sampling run is Ns + Nb
    # Na is not used in the old sampler, but hard-coded e.g. to int(0.1*Ns)
    np.random.seed(0)
    samples_old = sampler_old.sample_adapt(N=Ns, Nb=Nb).samples

    # Get Ns samples from the new sampler
    # Sampling run is Ns + Nb
    # Na is used in the new sampler, defined the "warmup" iterations
    # Nb+Ns-Na samples are taken from the new sampler after the warmup
    # Nb samples are removed afterwards as burn-in
    np.random.seed(0)
    samples_new = sampler_new.warmup(Na).sample(Nb+Ns-Na).get_samples().samples[...,Nb:-1]

    assert np.allclose(samples_old, samples_new, atol=atol), f"Old: {samples_old}\nNew: {samples_new}"

targets = [
    cuqi.testproblem.Deconvolution1D().posterior
]
""" List of targets to test against. """

# ============ MH ============

@pytest.mark.parametrize("target", targets)
def test_MH_regression_sample(target: cuqi.density.Density):
    """Test the MH sampler regression."""
    sampler_old = cuqi.sampler.MH(target, scale=1)
    sampler_new = cuqi.mcmc.MHNew(target, scale=1)
    assert_true_if_sampling_is_equivalent(sampler_old, sampler_new)

@pytest.mark.parametrize("target", targets)
def test_MH_regression_warmup(target: cuqi.density.Density):
    """Test the MH sampler regression."""
    sampler_old = cuqi.sampler.MH(target, scale=1)
    sampler_new = cuqi.mcmc.MHNew(target, scale=1)
    assert_true_if_warmup_is_equivalent(sampler_old, sampler_new)

# ============ pCN ============

@pytest.mark.parametrize("target", targets)
def test_pCN_regression_sample(target: cuqi.density.Density):
    """Test the pCN sampler regression."""
    sampler_old = cuqi.sampler.pCN(target, scale=1)
    sampler_new = cuqi.mcmc.pCNNew(target, scale=1)
    assert_true_if_sampling_is_equivalent(sampler_old, sampler_new)

@pytest.mark.parametrize("target", targets)
@pytest.mark.xfail(reason="The warmup is not equivalent at this point for pCN sampler.")
def test_pCN_regression_warmup(target: cuqi.density.Density):
    """Test the pCN sampler regression."""
    sampler_old = cuqi.sampler.pCN(target, scale=1)
    sampler_new = cuqi.mcmc.pCNNew(target, scale=1)
    assert_true_if_warmup_is_equivalent(sampler_old, sampler_new)

# ============ MALA ============

@pytest.mark.parametrize("target", targets)
def test_MALA_regression_sample(target: cuqi.density.Density):
    """Test the pCN sampler regression."""
    sampler_old = cuqi.sampler.MALA(target, scale=1)
    sampler_new = cuqi.mcmc.MALANew(target, scale=1)
    assert_true_if_sampling_is_equivalent(sampler_old, sampler_new)

@pytest.mark.parametrize("target", targets)
def test_MALA_regression_warmup(target: cuqi.density.Density):
    """Test the pCN sampler regression."""
    sampler_old = cuqi.sampler.MALA(target, scale=1)
    sampler_new = cuqi.mcmc.MALANew(target, scale=1)
    assert_true_if_warmup_is_equivalent(sampler_old, sampler_new)


# ============ Checkpointing ============

@pytest.mark.parametrize("sampler", [
    cuqi.mcmc.MALANew(cuqi.testproblem.Deconvolution1D().posterior, scale=1),
])
def test_checkpointing(sampler: cuqi.mcmc.SamplerNew):
    """ Check that the checkpointing functionality works. Tested with save_checkpoint(filename) and load_checkpoint(filename). """

    # Run sampler with some samples
    sampler.sample(100)

    # Save checkpoint
    sampler.save_checkpoint('checkpoint.pickle')

    # Reset (soft) the sampler, e.g. remove all samples but keep the state
    sampler.reset()

    # Do some more samples from pre-defined rng state
    np.random.seed(0)
    samples1 = sampler.sample(100).get_samples().samples

    # Now load the checkpoint on completely fresh sampler not even with target
    sampler_fresh = sampler.__class__(sampler.target) # In principle init with no arguments. Now still with target
    sampler_fresh.load_checkpoint('checkpoint.pickle')

    # Do some more samples from pre-defined rng state
    np.random.seed(0)
    samples2 = sampler_fresh.sample(100).get_samples().samples[...,:-1] # TODO. This needs to be fixed..

    # Check that the samples are the same
    assert np.allclose(samples1, samples2), f"Samples1: {samples1.samples}\nSamples2: {samples2.samples}"

