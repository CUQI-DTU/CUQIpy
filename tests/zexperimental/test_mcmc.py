import cuqi
import pytest
import numpy as np
import inspect
from numbers import Number

def assert_true_if_sampling_is_equivalent(
        sampler_old: cuqi.sampler.Sampler,
        sampler_new: cuqi.experimental.mcmc.SamplerNew,
        Ns=20, atol=1e-1, old_idx=[1, None], new_idx=[0, -1]):
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
        sampler_new: cuqi.experimental.mcmc.SamplerNew,
        Ns=20, Nb=20,
        strategy="MH_like", old_idx=[1, None], new_idx=[0, None]):
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
    elif strategy == "NUTS":
        tune_freq = 1/Nb
    else:
        raise NotImplementedError(f"Strategy {strategy} not implemented")

    # Get Ns samples from the old sampler
    # Sampling run is Ns + Nb
    # Tuning frequency parametrized but hard-coded, e.g. to int(0.1*Ns) for MH.
    np.random.seed(0)
    samples_old = sampler_old.sample_adapt(
        N=Ns, Nb=Nb).samples[...,old_idx[0]:old_idx[1]]

    # Get Ns samples from the new sampler
    # Sampling run is Ns + Nb
    # Tune_freq is used in the new sampler, defining how often to tune.
    # Nb samples are removed afterwards as burn-in
    np.random.seed(0)
    if strategy == "NUTS":
        sampler_new.warmup(Nb, tune_freq=tune_freq)
        sampler_new.sample(Ns=Ns-1)
        samples_new = sampler_new.get_samples().samples[...,new_idx[0]:new_idx[1]]
    else:
        sampler_new.warmup(Ns+Nb-1, tune_freq=tune_freq)
        samples_new = \
            sampler_new.get_samples().samples[...,Nb+new_idx[0]:new_idx[1]]

    assert np.allclose(samples_old, samples_new), f"Old: {samples_old[0]}\nNew: {samples_new[0]}"

targets = [
    cuqi.testproblem.Deconvolution1D(dim=2).posterior,
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

# ============ PCN ============

@pytest.mark.parametrize("target", targets)
def test_pCN_regression_sample(target: cuqi.density.Density):
    """Test the pCN sampler regression."""
    sampler_old = cuqi.sampler.pCN(target, scale=0.001)
    sampler_new = cuqi.experimental.mcmc.PCNNew(target, scale=0.001)
    assert_true_if_sampling_is_equivalent(sampler_old, sampler_new)

@pytest.mark.parametrize("target", targets)
def test_pCN_regression_warmup(target: cuqi.density.Density):
    """Test the pCN sampler regression."""
    sampler_old = cuqi.sampler.pCN(target, scale=0.001)
    sampler_new = cuqi.experimental.mcmc.PCNNew(target, scale=0.001)
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

# ============ LinearRTO ============

def create_multiple_likelihood_posterior_target(dim=16):
    """Create a target with multiple likelihoods."""
    A1, data1, info1 = cuqi.testproblem.Deconvolution1D(dim=dim, phantom='square').get_components()
    A2, data2, info2 = cuqi.testproblem.Deconvolution1D(dim=dim, phantom='square').get_components()

    x = cuqi.distribution.Gaussian(0.5*np.ones(dim), 0.1)
    y1 = cuqi.distribution.Gaussian(A1@x, 0.001)
    y2 = cuqi.distribution.Gaussian(A2@x, 0.001)

    target = cuqi.distribution.JointDistribution(x,y1,y2)(y1=data1, y2=data2)

    return target

LinearRTO_targets = targets + [
    create_multiple_likelihood_posterior_target(dim=32),
    create_multiple_likelihood_posterior_target(dim=64),
    create_multiple_likelihood_posterior_target(dim=128)
]

@pytest.mark.parametrize("target", LinearRTO_targets)
def test_LinearRTO_regression_sample(target: cuqi.density.Density):
    """Test the LinearRTO sampler regression."""
    sampler_old = cuqi.sampler.LinearRTO(target)
    sampler_new = cuqi.experimental.mcmc.LinearRTONew(target)
    assert_true_if_sampling_is_equivalent(sampler_old, sampler_new)

@pytest.mark.parametrize("target", LinearRTO_targets)
def test_LinearRTO_regression_warmup(target: cuqi.density.Density):
    """Test the LinearRTO sampler regression."""
    sampler_old = cuqi.sampler.LinearRTO(target)
    sampler_new = cuqi.experimental.mcmc.LinearRTONew(target)
    assert_true_if_warmup_is_equivalent(sampler_old, sampler_new)

# ============ RegularizedLinearRTO ============

def create_regularized_target(dim=16):
    """Create a regularized target."""
    A, y_data, info = cuqi.testproblem.Deconvolution1D(dim=dim, phantom='square').get_components()
    x = cuqi.implicitprior.RegularizedGaussian(0.5*np.ones(dim), 0.1, constraint = "nonnegativity")
    y = cuqi.distribution.Gaussian(A@x, 0.001)
    return cuqi.distribution.JointDistribution(x, y)(y=y_data)

def create_multiple_likelihood_posterior_regularized_target(dim=16):
    """Create a target with multiple likelihoods and a regularized prior."""
    A1, data1, info1 = cuqi.testproblem.Deconvolution1D(dim=dim, phantom='square').get_components()
    A2, data2, info2 = cuqi.testproblem.Deconvolution1D(dim=dim, phantom='square').get_components()

    x = cuqi.implicitprior.RegularizedGaussian(0.5*np.ones(dim), 0.1, constraint = "nonnegativity")
    y1 = cuqi.distribution.Gaussian(A1@x, 0.001)
    y2 = cuqi.distribution.Gaussian(A2@x, 0.001)

    target = cuqi.distribution.JointDistribution(x,y1,y2)(y1=data1, y2=data2)

    return target

regularized_targets = [
    create_regularized_target(dim=16),
    create_regularized_target(dim=128)
] + [
    create_multiple_likelihood_posterior_regularized_target(dim=16),
    create_multiple_likelihood_posterior_regularized_target(dim=128)
]

@pytest.mark.parametrize("target", regularized_targets)
def test_RegularizedLinearRTO_regression_sample(target: cuqi.density.Density):
    """Test the RegularizedLinearRTO sampler regression."""
    sampler_old = cuqi.sampler.RegularizedLinearRTO(target, stepsize=1e-3)
    sampler_new = cuqi.experimental.mcmc.RegularizedLinearRTONew(target, stepsize=1e-3)
    assert_true_if_sampling_is_equivalent(sampler_old, sampler_new)

@pytest.mark.parametrize("target", regularized_targets)
def test_RegularizedLinearRTO_regression_warmup(target: cuqi.density.Density):
    """Test the RegularizedLinearRTO sampler regression."""

    sampler_old = cuqi.sampler.RegularizedLinearRTO(target, stepsize=1e-3)
    sampler_new = cuqi.experimental.mcmc.RegularizedLinearRTONew(target, stepsize=1e-3)
    assert_true_if_warmup_is_equivalent(sampler_old, sampler_new)

def create_lmrf_prior_target(dim=16):
    """Create a target with LMRF prior."""
    A, y_data, info = cuqi.testproblem.Deconvolution1D(dim=dim, phantom='square').get_components()
    x = cuqi.distribution.LMRF(0, 0.1, geometry=dim)
    y = cuqi.distribution.Gaussian(A@x, 0.001)
    return cuqi.distribution.JointDistribution(x, y)(y=y_data)



@pytest.mark.parametrize("target_dim", [16, 128])
def test_UGLA_regression_sample(target_dim):
    """Test the UGLA sampler regression."""
    target = create_lmrf_prior_target(dim=target_dim)
    sampler_old = cuqi.sampler.UGLA(target)
    sampler_new = cuqi.experimental.mcmc.UGLANew(target)
    assert_true_if_sampling_is_equivalent(sampler_old, sampler_new)

@pytest.mark.parametrize("target_dim", [16, 128])
def test_UGLA_regression_warmup(target_dim):
    """Test the UGLA sampler regression."""
    target = create_lmrf_prior_target(dim=target_dim)
    sampler_old = cuqi.sampler.UGLA(target)
    sampler_new = cuqi.experimental.mcmc.UGLANew(target)
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
                                          old_idx=[1, -1],
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
                                        old_idx=[1, -1],
                                        new_idx=[1, None])

# ============= HMC (NUTS) ==============
@pytest.mark.parametrize("target", targets)
def test_NUTS_regression_sample(target: cuqi.density.Density):
    """Test the HMC (NUTS) sampler regression."""
    sampler_old = cuqi.sampler.NUTS(target, adapt_step_size=0.001)
    sampler_new = cuqi.experimental.mcmc.NUTSNew(target, step_size=0.001)
    assert_true_if_sampling_is_equivalent(sampler_old, sampler_new, Ns=20)

@pytest.mark.parametrize("target", targets)
def test_NUTS_regression_sample_tune_first_step_only(
    target: cuqi.density.Density):
    """Test the HMC (NUTS) sampler regression."""
    sampler_old = cuqi.sampler.NUTS(target, adapt_step_size=False)
    sampler_new = cuqi.experimental.mcmc.NUTSNew(target, step_size=None)
    assert_true_if_sampling_is_equivalent(sampler_old, sampler_new, Ns=20)

@pytest.mark.parametrize("target", targets)
def test_NUTS_regression_warmup(target: cuqi.density.Density):
    """Test the HMC (NUTS) sampler regression (with warmup)."""
    sampler_old = cuqi.sampler.NUTS(target, adapt_step_size=True)
    sampler_old._return_burnin = True
    sampler_new = cuqi.experimental.mcmc.NUTSNew(target, step_size=None)
    Ns = 20
    Nb = 20
    assert_true_if_warmup_is_equivalent(sampler_old,
                                        sampler_new,
                                        Ns=Ns,
                                        Nb=Nb,
                                        strategy="NUTS")
# ============ Checkpointing ============


# List of all samplers from cuqi.experimental.mcmc that should be tested for checkpointing + their parameters
checkpoint_targets = [
    cuqi.experimental.mcmc.ULANew(cuqi.testproblem.Deconvolution1D().posterior, scale=0.0001),
    cuqi.experimental.mcmc.MALANew(cuqi.testproblem.Deconvolution1D().posterior, scale=0.0001),
    cuqi.experimental.mcmc.LinearRTONew(cuqi.testproblem.Deconvolution1D().posterior),
    cuqi.experimental.mcmc.UGLANew(create_lmrf_prior_target(dim=16))
]
    
# List of samplers from cuqi.experimental.mcmc that should be skipped for checkpoint testing
skip_checkpoint = [
    cuqi.experimental.mcmc.SamplerNew,
    cuqi.experimental.mcmc.ProposalBasedSamplerNew,
    cuqi.experimental.mcmc.MHNew,
    cuqi.experimental.mcmc.PCNNew,
    cuqi.experimental.mcmc.CWMHNew,
    cuqi.experimental.mcmc.RegularizedLinearRTONew, # Due to the _choose_stepsize method
    cuqi.experimental.mcmc.NUTSNew,
]

def test_ensure_all_not_skipped_samplers_are_tested_for_checkpointing():
    """Ensure that all samplers from cuqi.experimental.mcmc, except those skipped, are tested for checkpointing."""

    # List of all samplers from cuqi.experimental.mcmc that should be tested for checkpointing
    samplers = [
        cls
        for _, cls in inspect.getmembers(cuqi.experimental.mcmc, inspect.isclass)
        if cls not in skip_checkpoint  # use cls here, not name
    ]

    # Convert instances in checkpoint_targets to their classes
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

    # Reset the sampler, e.g. remove all samples but keep the state
    # Calling sampler.reset() would fail since it resets state.
    # Instead, we want to call a method like reset_history().
    # Currently we do this:
    sampler._samples = []

    # Do some more samples from pre-defined rng state
    np.random.seed(0)
    samples1 = sampler.warmup(50).sample(50).get_samples().samples

    #TODO Consider changing now that target=None is allowed.
    # Now load the checkpoint on completely fresh sampler not even with target
    sampler_fresh = sampler.__class__(sampler.target) # In principle init with no arguments. Now still with target
    sampler_fresh.load_checkpoint('checkpoint.pickle')

    # Do some more samples from pre-defined rng state
    np.random.seed(0)
    samples2 = sampler_fresh.warmup(50).sample(50).get_samples().samples

    # Check that the samples are the same
    assert np.allclose(samples1, samples2), f"Samples1: {samples1}\nSamples2: {samples2}"


state_history_targets = [
    cuqi.experimental.mcmc.MHNew(cuqi.testproblem.Deconvolution1D(dim=10).posterior, scale=0.0001),
    cuqi.experimental.mcmc.PCNNew(cuqi.testproblem.Deconvolution1D(dim=10).posterior, scale=0.001),
    cuqi.experimental.mcmc.CWMHNew(cuqi.testproblem.Deconvolution1D(dim=10).posterior, scale=0.001),
    cuqi.experimental.mcmc.ULANew(cuqi.testproblem.Deconvolution1D(dim=10).posterior, scale=0.0001),
    cuqi.experimental.mcmc.MALANew(cuqi.testproblem.Deconvolution1D(dim=10).posterior, scale=0.0001),
    cuqi.experimental.mcmc.LinearRTONew(cuqi.testproblem.Deconvolution1D(dim=10).posterior),
    cuqi.experimental.mcmc.RegularizedLinearRTONew(create_regularized_target(dim=16)),
    cuqi.experimental.mcmc.UGLANew(create_lmrf_prior_target(dim=32)),
]


@pytest.mark.parametrize("sampler", state_history_targets)
def test_state_keys(sampler: cuqi.experimental.mcmc.SamplerNew):
    """Test that the state keys match the expected keys defined in _STATE_KEYS."""

    # Run sampler to initialize state variables
    sampler.warmup(10).sample(10)
    
    # Retrieve the state of the sampler
    state = sampler.get_state() # Will fail if variable for state is not set in the sampler

    # Retrieve the actual keys from the saved state
    actual_keys = set(state['state'].keys())

    # Retrieve the expected keys from the sampler's _STATE_KEYS
    expected_keys = set(sampler._STATE_KEYS)

    # Check if the actual keys match the expected keys
    assert actual_keys == expected_keys, f"State keys mismatch. Expected: {expected_keys}, Actual: {actual_keys}"

@pytest.mark.parametrize("sampler", state_history_targets)
def test_history_keys(sampler: cuqi.experimental.mcmc.SamplerNew):
    """Test that the history keys match the expected keys defined in _HISTORY_KEYS."""

    # Run sampler to initialize history variables
    sampler.warmup(10).sample(10)
    
    # Retrieve the history of the sampler
    history = sampler.get_history() # Will fail if variable for history is not set in the sampler

    # Retrieve the actual keys from the saved history
    actual_keys = set(history['history'].keys())

    # Retrieve the expected keys from the sampler's _HISTORY_KEYS
    expected_keys = set(sampler._HISTORY_KEYS)

    # Check if the actual keys match the expected keys
    assert actual_keys == expected_keys, f"History keys mismatch. Expected: {expected_keys}, Actual: {actual_keys}"

# Dictionary to store keys that are not expected to be updated after warmup.
# Likely due to not implemented feature in the sampler.
state_exception_keys = {
    cuqi.experimental.mcmc.ULANew: 'scale',
    cuqi.experimental.mcmc.MALANew: 'scale',
}

@pytest.mark.parametrize("sampler", state_history_targets)
def test_state_is_fully_updated_after_warmup_step(sampler: cuqi.experimental.mcmc.SamplerNew):
    """ Test that the state is fully updated after a warmup step.
    
    This also checks that the samplers use (or at least update) all the keys defined in _STATE_KEYS.

    """

    # Extract the initial state of the sampler
    initial_state = sampler.get_state()

    # Run 100 warmup steps (should be enough to update all state variables)
    sampler.warmup(100)

    # Extract the state of the sampler after the warmup step
    updated_state = sampler.get_state()

    # Dictionary to store messages for keys that have not been updated
    failed_updates = {}

    # Ensure all keys in _STATE_KEYS are present in the state and have been updated
    for key in sampler._STATE_KEYS:

        # Skip keys that are not expected to be updated after warmup
        if key in state_exception_keys.get(sampler.__class__, []):
            continue

        initial_value = initial_state['state'].get(key)
        updated_value = updated_state['state'].get(key)

        # Check all state variables are updated after warmup
        if isinstance(initial_value, np.ndarray) and isinstance(updated_value, np.ndarray):
            if np.allclose(updated_value, initial_value):
                failed_updates[key] = f"(Arrays are equal)"
        else:
            if updated_value == initial_value:
                failed_updates[key] = f"Initial: {initial_value}, Updated: {updated_value}"

    # Assert that there were no errors during the state checks
    if failed_updates:
        failed_keys = ', '.join(failed_updates.keys())
        error_details = '\n'.join([f"State '{key}' not updated correctly after warmup. {message}" for key, message in failed_updates.items()])
        error_message = f"Errors occurred in {sampler.__class__.__name__} - issues with keys: {failed_keys}.\n{error_details}"
        assert not failed_updates, error_message

# Samplers that should be tested for target=None initialization
initialize_testing_sampler_classes = [
    cls
    for _, cls in inspect.getmembers(cuqi.experimental.mcmc, inspect.isclass)
    if cls not in [cuqi.experimental.mcmc.SamplerNew, cuqi.experimental.mcmc.ProposalBasedSamplerNew]
]

# Instances of samplers that should be tested for target=None initialization consistency
initialize_testing_sampler_instances = [
    cuqi.experimental.mcmc.MHNew(target=cuqi.testproblem.Deconvolution1D(dim=10).posterior),
    cuqi.experimental.mcmc.PCNNew(target=cuqi.testproblem.Deconvolution1D(dim=10).posterior),
    cuqi.experimental.mcmc.CWMHNew(target=cuqi.testproblem.Deconvolution1D(dim=10).posterior),
    cuqi.experimental.mcmc.ULANew(target=cuqi.testproblem.Deconvolution1D(dim=10).posterior),
    cuqi.experimental.mcmc.MALANew(target=cuqi.testproblem.Deconvolution1D(dim=10).posterior),
    cuqi.experimental.mcmc.NUTSNew(target=cuqi.testproblem.Deconvolution1D(dim=10).posterior),
    cuqi.experimental.mcmc.LinearRTONew(target=cuqi.testproblem.Deconvolution1D(dim=10).posterior),
    cuqi.experimental.mcmc.RegularizedLinearRTONew(target=create_regularized_target(dim=16)),
    cuqi.experimental.mcmc.UGLANew(target=create_lmrf_prior_target(dim=16)),
]


@pytest.mark.parametrize("sampler_class", initialize_testing_sampler_classes)
def test_target_None_init_in_samplers(sampler_class):
    """ Test all samplers can be initialized with target=None. """
    sampler = sampler_class(target=None)
    assert sampler.target is None, f"Sampler {sampler_class} failed to initialize with target=None"

@pytest.mark.parametrize("sampler_class", initialize_testing_sampler_classes)
def test_sampler_initialization_consistency(sampler_class: cuqi.experimental.mcmc.SamplerNew):
    """ Test that all samplers initialized with target=None and target set later is equivalent to initializing with target right away. """

    # Find sampler instance that matches the sampler class
    sampler_instance = next((s for s in initialize_testing_sampler_instances if isinstance(s, sampler_class)), None)
    if sampler_instance is None:
        raise ValueError(
            f"No sampler instance in the list of initialize_testing_sampler_instances matches the sampler class {sampler_class}. "
            "Please add an instance to the list."
        )
    
    # Initialize sampler with target=None and set target after
    sampler_target_None = sampler_class(target=None)
    sampler_target_None.target = sampler_instance.target

    # Then run samplers and compare the samples
    np.random.seed(0)
    samples_target_None = sampler_target_None.warmup(10).sample(10).get_samples().samples

    np.random.seed(0)
    samples_target = sampler_instance.warmup(10).sample(10).get_samples().samples

    assert np.allclose(samples_target_None, samples_target), f"Sampler {sampler_class} initialized with target=None is not equivalent to initializing with target right away."

def compare_attributes(attr1, attr2, key=''):
    """ Recursively compare attributes. """
    try:
        if isinstance(attr1, np.ndarray):
            assert np.allclose(attr1, attr2), f"ndarray: Attribute '{key}' differs after initialization"
        elif isinstance(attr1, list):
            assert len(attr1) == len(attr2), f"List length: Attribute '{key}' differs after initialization"
            for i, (a1, a2) in enumerate(zip(attr1, attr2)):
                compare_attributes(a1, a2, f"{key}[{i}]")
        elif isinstance(attr1, dict):
            assert attr1.keys() == attr2.keys(), f"Dict keys: Attribute '{key}' differs after initialization"
            for k in attr1.keys():
                compare_attributes(attr1[k], attr2[k], f"{key}['{k}']")
        elif isinstance(attr1, Number):
            assert np.allclose(attr1, attr2), f"Number: Attribute '{key}' differs after initialization"
        elif isinstance(attr1, str):
            assert attr1 == attr2, f"String: Attribute '{key}' differs after initialization"
        else:
            assert type(attr1) == type(attr2), f"Type: Attribute '{key}' differs after initialization"
    except AssertionError as e:
        raise AssertionError(f"{e}. Ensure the attribute '{key}' is correctly defined and initialized in the _initialize method.")

@pytest.mark.parametrize("sampler_class", initialize_testing_sampler_classes)
def test_sampler_reinitialization_restores_to_initial_configuration(sampler_class):

    # Find sampler instance that matches the sampler class
    sampler_instance = next((s for s in initialize_testing_sampler_instances if isinstance(s, sampler_class)), None)
    if (sampler_instance is None):
        raise ValueError(
            f"No instance in the list of initialize_testing_sampler_instances matches the class {sampler_class}. "
            "Please add an instance to the list."
        )
    
    # Initialize two instances with equivalent initial configuration
    instance1 = sampler_class(target=sampler_instance.target)
    instance2 = sampler_class(target=sampler_instance.target)

    # Run instance1 for a bit. Then reinitialize it and initialize instance2
    instance1.warmup(10).sample(10)
    instance1.reinitialize()
    instance2.initialize()

    # Compare all attributes of the two instances. They should be equivalent.
    for key in instance1.__dict__.keys() | instance2.__dict__.keys():
        attr1 = getattr(instance1, key, None)
        attr2 = getattr(instance2, key, None)
        compare_attributes(attr1, attr2, key)
