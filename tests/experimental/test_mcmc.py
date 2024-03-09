import cuqi
import pytest
import numpy as np

def assert_true_if_sampling_is_equivalent(
        sampler_old: cuqi.sampler.Sampler,
        sampler_new: cuqi.experimental.mcmc.SamplerNew,
        Ns=100, atol=1e-1, old_idx=[0, None], new_idx=[0, -1]):
    """ Assert that the samples from the old and new sampler are equivalent. """

    np.random.seed(0)
    samples_old = sampler_old.sample(Ns).samples[..., old_idx[0]:old_idx[1]]

    np.random.seed(0)
    samples_new = (
        sampler_new.sample(Ns).get_samples().samples[..., new_idx[0]:new_idx[1]]
    )

    assert np.allclose(samples_old, samples_new, atol=atol), f"Old: {samples_old}\nNew: {samples_new}"

def assert_true_if_warmup_is_equivalent(
        sampler_old: cuqi.sampler.Sampler,
        sampler_new: cuqi.experimental.mcmc.SamplerNew, Ns=100, Nb=100,
        strategy="MH_like", old_idx=[0, None], new_idx=[0, None]):
    """ Assert that the samples from the old and new sampler are equivalent.
     
    Ns: int
        Number of samples.
    
    Nb: int
        Number of burn-in samples. (to be removed from the samples)

    strategy: str
        Tuning strategy defined by sampler to compare with. Default is MH.
              
    """

    if strategy == "MH_like":
        tune_freq = int(0.1*Ns) / (Ns+Nb-1) # Due to a bug? in old MH, tuning freq is only defined by N and not N+Nb.
    else:
        raise NotImplementedError(f"Strategy {strategy} not implemented")

    # Get Ns samples from the old sampler
    # Sampling run is Ns + Nb
    # Tuning frequency parametrized but hard-coded, e.g. to int(0.1*Ns) for MH.
    np.random.seed(0)
    samples_old =\
        sampler_old.sample_adapt(N=Ns, Nb=Nb).samples[...,old_idx[0]:old_idx[1]]

    # Get Ns samples from the new sampler
    # Sampling run is Ns + Nb
    # Tune_freq is used in the new sampler, defining how often to tune.
    # Nb samples are removed afterwards as burn-in
    np.random.seed(0)    
    samples_new = sampler_new.warmup(Ns+Nb-1, tune_freq=tune_freq).get_samples().samples[...,Nb+new_idx[0]:new_idx[1]]

    assert np.allclose(samples_old, samples_new), f"Old: {samples_old[0]}\nNew: {samples_new[0]}"

targets = [
    cuqi.testproblem.Deconvolution1D(dim=2).posterior,
    cuqi.testproblem.Deconvolution1D(dim=20).posterior,
    cuqi.testproblem.Deconvolution1D(dim=128).posterior
]
""" List of targets to test against. """


def assert_true_if_warmup_is_equivalent2(
        sampler_old: cuqi.sampler.Sampler, sampler_new: cuqi.mcmc.SamplerNew):
    """ Assert that the samples from the old and new sampler are equivalent.
     
    Ns: int
        Number of samples.
    
    Nb: int
        Number of burn-in samples. (to be removed from the samples)

    Na: int
        Number of iterations to adapt.
              
    """

    np.random.seed(0)
    sampler_old.sample_adapt(N=101)

    np.random.seed(0)
    sampler_new.warmup(100)
    

    assert np.allclose(sampler_old.scale, sampler_new.scale)

# ============ MH ============

@pytest.mark.parametrize("target", targets)
def test_MH_regression_sample(target: cuqi.density.Density):
    """Test the MH sampler regression."""
    sampler_old = cuqi.sampler.MH(target, scale=1)
    sampler_new = cuqi.experimental.mcmc.MHNew(target, scale=1)
    assert_true_if_sampling_is_equivalent(sampler_old, sampler_new)

@pytest.mark.parametrize("target", targets)
def test_MH_regression_warmup(target: cuqi.density.Density):
    """Test the MH sampler regression."""
    sampler_old = cuqi.sampler.MH(target, scale=1)
    sampler_new = cuqi.experimental.mcmc.MHNew(target, scale=1)
    assert_true_if_warmup_is_equivalent(sampler_old, sampler_new)

# ============ pCN ============

@pytest.mark.parametrize("target", targets)
def test_pCN_regression_sample(target: cuqi.density.Density):
    """Test the pCN sampler regression."""
    sampler_old = cuqi.sampler.pCN(target, scale=1)
    sampler_new = cuqi.experimental.mcmc.pCNNew(target, scale=1)
    assert_true_if_sampling_is_equivalent(sampler_old, sampler_new)

@pytest.mark.parametrize("target", targets)
@pytest.mark.xfail(reason="The warmup is not equivalent at this point for pCN sampler.")
def test_pCN_regression_warmup(target: cuqi.density.Density):
    """Test the pCN sampler regression."""
    sampler_old = cuqi.sampler.pCN(target, scale=1)
    sampler_new = cuqi.experimental.mcmc.pCNNew(target, scale=1)
    assert_true_if_warmup_is_equivalent(sampler_old, sampler_new)

# ============ MALA ============

@pytest.mark.parametrize("target", targets)
def test_MALA_regression_sample(target: cuqi.density.Density):
    """Test the pCN sampler regression."""
    sampler_old = cuqi.sampler.MALA(target, scale=1)
    sampler_new = cuqi.experimental.mcmc.MALANew(target, scale=1)
    assert_true_if_sampling_is_equivalent(sampler_old, sampler_new)

@pytest.mark.parametrize("target", targets)
def test_MALA_regression_warmup(target: cuqi.density.Density):
    """Test the pCN sampler regression."""
    sampler_old = cuqi.sampler.MALA(target, scale=1)
    sampler_new = cuqi.experimental.mcmc.MALANew(target, scale=1)
    assert_true_if_warmup_is_equivalent(sampler_old, sampler_new)

# # ============ CWMH ============

@pytest.mark.parametrize("target", targets)
def test_CWMH_regression_sample(target: cuqi.density.Density):
    """Test the CWMH sampler regression."""
    sampler_old = cuqi.sampler.CWMH(target,
                                    scale=np.ones(target.dim),
                                    x0=np.zeros(target.dim))
    sampler_new = cuqi.mcmc.CWMHNew(target, scale=np.ones(target.dim),
                                    initial_point=np.zeros(target.dim))
    assert_true_if_sampling_is_equivalent(sampler_old, sampler_new,
                                          Ns=10,
                                          old_idx=[0, -1],
                                          new_idx=[1, -1])

@pytest.mark.parametrize("target", targets)
def test_CWMH_regression_warmup2(target: cuqi.density.Density):
    """Test the CWMH sampler regression."""
    sampler_old = cuqi.sampler.CWMH(target, scale=np.ones(target.dim))
    sampler_new = cuqi.mcmc.CWMHNew(target, scale=np.ones(target.dim))
    assert_true_if_warmup_is_equivalent2(sampler_old, sampler_new)

@pytest.mark.parametrize("target", targets)
def test_CWMH_regression_warmup(target: cuqi.density.Density):
    """Test the CWMH sampler regression."""
    sampler_old = cuqi.sampler.CWMH(target, scale=np.ones(target.dim))
    sampler_new = cuqi.mcmc.CWMHNew(target, scale=np.ones(target.dim))
    assert_true_if_warmup_is_equivalent(sampler_old, sampler_new,
                                        Ns=100,
                                        strategy="MH_like",
                                        old_idx=[0, -1],
                                        new_idx=[1, None])

# ============ Checkpointing ============

@pytest.mark.parametrize("sampler", [
    cuqi.experimental.mcmc.MALANew(cuqi.testproblem.Deconvolution1D().posterior, scale=1),
])
def test_checkpointing(sampler: cuqi.experimental.mcmc.SamplerNew):
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
