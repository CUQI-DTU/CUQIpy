import cuqi
import pytest
import numpy as np
import inspect

def assert_true_if_sampling_is_equivalent(
        sampler_old: cuqi.sampler.Sampler,
        sampler_new: cuqi.experimental.mcmc.SamplerNew,
        Ns=100, atol=1e-1, old_idx=[0, None], new_idx=[0, -1]):
    """ Assert that the samples from the old and new sampler are equivalent.

    Ns: int
        Number of samples.

    Nb: int
        Number of burn-in samples. (to be removed from the samples)

    old_idx: list of length 2
        Indexes to slice the samples from the old sampler. The first index is
        the index of the first sample to be compared and the second index is of
        the last sample.

    new_idx: list of length 2
        Indexes to slice the samples from the new sampler. The first index is
        the index of the first sample to be compared and the second index is of
        the last sample.
    """
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

    old_idx: list of length 2
        Indexes to slice the samples from the old sampler. The first index is
        the index of the first sample to be compared and the second index is of
        the last sample.

    new_idx: list of length 2
        Indexes to slice the samples from the new sampler. The first index is
        the index of the first sample to be compared (after burn-in is removed,
        i.e. samples start index is Nb+new_idx[0]) and the second index is of
        the last sample.
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
    sampler_new.warmup(Ns+Nb-1, tune_freq=tune_freq)  
    samples_new = \
        sampler_new.get_samples().samples[...,Nb+new_idx[0]:new_idx[1]]

    assert np.allclose(samples_old, samples_new), f"Old: {samples_old[0]}\nNew: {samples_new[0]}"

targets = [
    cuqi.testproblem.Deconvolution1D(dim=2).posterior,
    cuqi.testproblem.Deconvolution1D(dim=20).posterior,
    cuqi.testproblem.Deconvolution1D(dim=128).posterior
]
""" List of targets to test against. """

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

# ============ ULA ============

@pytest.mark.parametrize("target", targets)
def test_ULA_regression_sample(target: cuqi.density.Density):
    """Test the ULA sampler regression."""
    sampler_old = cuqi.sampler.ULA(target, scale=0.1)
    sampler_new = cuqi.experimental.mcmc.ULANew(target, scale=0.1)
    assert_true_if_sampling_is_equivalent(sampler_old, sampler_new)

@pytest.mark.parametrize("target", targets)
def test_ULA_regression_warmup(target: cuqi.density.Density):
    """Test the ULA sampler regression."""
    sampler_old = cuqi.sampler.ULA(target, scale=0.001)
    sampler_new = cuqi.experimental.mcmc.ULANew(target, scale=0.001)
    assert_true_if_warmup_is_equivalent(sampler_old, sampler_new)

# ============ MALA ============

@pytest.mark.parametrize("target", targets)
def test_MALA_regression_sample(target: cuqi.density.Density):
    """Test the MALA sampler regression."""
    sampler_old = cuqi.sampler.MALA(target, scale=1)
    sampler_new = cuqi.experimental.mcmc.MALANew(target, scale=1)
    assert_true_if_sampling_is_equivalent(sampler_old, sampler_new)

@pytest.mark.parametrize("target", targets)
def test_MALA_regression_warmup(target: cuqi.density.Density):
    """Test the MALA sampler regression."""
    sampler_old = cuqi.sampler.MALA(target, scale=1)
    sampler_new = cuqi.experimental.mcmc.MALANew(target, scale=1)
    assert_true_if_warmup_is_equivalent(sampler_old, sampler_new)

# ============== CWMH ============

@pytest.mark.parametrize("target", targets)
def test_CWMH_regression_sample(target: cuqi.density.Density):
    """Test the CWMH sampler regression."""
    sampler_old = cuqi.sampler.CWMH(target, scale=np.ones(target.dim))
    sampler_new = cuqi.experimental.mcmc.CWMHNew(target,
                                                 scale=np.ones(target.dim))
    assert_true_if_sampling_is_equivalent(sampler_old, sampler_new,
                                          Ns=10,
                                          old_idx=[0, -1],
                                          new_idx=[1, -1])

@pytest.mark.parametrize("target", targets)
def test_CWMH_regression_warmup(target: cuqi.density.Density):
    """Test the CWMH sampler regression."""
    sampler_old = cuqi.sampler.CWMH(target, scale=np.ones(target.dim))
    sampler_new = cuqi.experimental.mcmc.CWMHNew(target,
                                                 scale=np.ones(target.dim))
    Ns = 100 if target.dim < 50 else 20
    assert_true_if_warmup_is_equivalent(sampler_old, sampler_new,
                                        Ns=Ns,
                                        strategy="MH_like",
                                        old_idx=[0, -1],
                                        new_idx=[1, None])

# ============ Checkpointing ============


# List of all samplers from cuqi.experimental.mcmc that should be tested for checkpointing + their parameters
checkpoint_targets = [
    cuqi.experimental.mcmc.ULANew(cuqi.testproblem.Deconvolution1D().posterior, scale=0.0001),
    cuqi.experimental.mcmc.MALANew(cuqi.testproblem.Deconvolution1D().posterior, scale=0.0001),
]
    
# List of samplers from cuqi.experimental.mcmc that should be skipped for checkpoint testing
skip_checkpoint = [
    cuqi.experimental.mcmc.SamplerNew,
    cuqi.experimental.mcmc.ProposalBasedSamplerNew,
    cuqi.experimental.mcmc.MHNew,
    cuqi.experimental.mcmc.pCNNew,
    cuqi.experimental.mcmc.CWMHNew
]

def test_ensure_all_not_skipped_samplers_are_tested_for_checkpointing():
    """Ensure that all samplers from cuqi.experimental.mcmc, except those skipped, are tested for checkpointing."""

    # List of all samplers from cuqi.experimental.mcmc that should be tested for checkpointing
    samplers = [
        cls
        for _, cls in inspect.getmembers(cuqi.experimental.mcmc, inspect.isclass)
        if cls not in skip_checkpoint  # use cls here, not name
    ]

    # Convert instances to their classes
    checkpoint_target_classes = [type(sampler) for sampler in checkpoint_targets]  

    # Convert 'samplers' classes to names for easier comparison and error reading
    sampler_names = [cls.__name__ for cls in samplers]
    
    # 'checkpoint_target_classes' already contains classes, convert them to names
    checkpoint_target_names = [cls.__name__ for cls in checkpoint_target_classes]
    
    # Now, assert that sets of names match
    assert set(sampler_names) == set(checkpoint_target_names), f"Samplers not tested for checkpointing: {set(sampler_names) - set(checkpoint_target_names)}"


@pytest.mark.parametrize("sampler", checkpoint_targets)
def test_checkpointing(sampler: cuqi.experimental.mcmc.SamplerNew):
    """ Check that the checkpointing functionality works. Tested with save_checkpoint(filename) and load_checkpoint(filename).
    This also implicitly tests the get_state(), set_state(), get_history(), and set_history() as well as the reset() methods.
    
    """

    # Run sampler with some samples
    sampler.warmup(50).sample(50)

    # Save checkpoint
    sampler.save_checkpoint('checkpoint.pickle')

    # Reset (soft) the sampler, e.g. remove all samples but keep the state
    sampler.reset()

    # Do some more samples from pre-defined rng state
    np.random.seed(0)
    samples1 = sampler.warmup(50).sample(50).get_samples().samples

    # Now load the checkpoint on completely fresh sampler not even with target
    sampler_fresh = sampler.__class__(sampler.target) # In principle init with no arguments. Now still with target
    sampler_fresh.load_checkpoint('checkpoint.pickle')

    # Do some more samples from pre-defined rng state
    np.random.seed(0)
    samples2 = sampler_fresh.warmup(50).sample(50).get_samples().samples[...,1:] # TODO. This needs to be fixed.. We should likely not store initial point in _samples

    # Check that the samples are the same
    assert np.allclose(samples1, samples2), f"Samples1: {samples1}\nSamples2: {samples2}"


@pytest.mark.parametrize("sampler", checkpoint_targets)
def test_state_keys(sampler: cuqi.experimental.mcmc.SamplerNew):
    """Test that the state keys match the expected keys defined in _STATE_KEYS."""

    # Run sampler to initialize state variables
    sampler.warmup(10).sample(10)
    
    # Retrieve the state of the sampler
    state = sampler.get_state()

    # Retrieve the actual keys from the saved state
    actual_keys = set(state['state'].keys())

    # Retrieve the expected keys from the sampler's _STATE_KEYS
    expected_keys = set(sampler._STATE_KEYS)

    # Check if the actual keys match the expected keys
    assert actual_keys == expected_keys, f"State keys mismatch. Expected: {expected_keys}, Actual: {actual_keys}"

@pytest.mark.parametrize("sampler", checkpoint_targets)
def test_history_keys(sampler: cuqi.experimental.mcmc.SamplerNew):
    """Test that the history keys match the expected keys defined in _HISTORY_KEYS."""

    # Run sampler to initialize history variables
    sampler.warmup(10).sample(10)
    
    # Retrieve the history of the sampler
    history = sampler.get_history()

    # Retrieve the actual keys from the saved history
    actual_keys = set(history['history'].keys())

    # Retrieve the expected keys from the sampler's _HISTORY_KEYS
    expected_keys = set(sampler._HISTORY_KEYS)

    # Check if the actual keys match the expected keys
    assert actual_keys == expected_keys, f"History keys mismatch. Expected: {expected_keys}, Actual: {actual_keys}"
