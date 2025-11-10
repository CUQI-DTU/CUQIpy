import cuqi
import pytest
import numpy as np
import inspect
from numbers import Number

def assert_true_if_sampling_is_equivalent(
        sampler_old: cuqi.legacy.sampler.Sampler,
        sampler_new: cuqi.sampler.Sampler,
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
        sampler_old: cuqi.legacy.sampler.Sampler,
        sampler_new: cuqi.sampler.Sampler,
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
    sampler_old = cuqi.legacy.sampler.MH(target, scale=1)
    sampler_new = cuqi.sampler.MH(target, scale=1)
    assert_true_if_sampling_is_equivalent(sampler_old, sampler_new)

@pytest.mark.parametrize("target", targets)
def test_MH_regression_warmup(target: cuqi.density.Density):
    """Test the MH sampler regression."""
    sampler_old = cuqi.legacy.sampler.MH(target, scale=1)
    sampler_new = cuqi.sampler.MH(target, scale=1)
    assert_true_if_warmup_is_equivalent(sampler_old, sampler_new)

# ============ PCN ============

@pytest.mark.parametrize("target", targets)
def test_pCN_regression_sample(target: cuqi.density.Density):
    """Test the pCN sampler regression."""
    sampler_old = cuqi.legacy.sampler.pCN(target, scale=0.001)
    sampler_new = cuqi.sampler.PCN(target, scale=0.001)
    assert_true_if_sampling_is_equivalent(sampler_old, sampler_new)

@pytest.mark.parametrize("target", targets)
def test_pCN_regression_warmup(target: cuqi.density.Density):
    """Test the pCN sampler regression."""
    sampler_old = cuqi.legacy.sampler.pCN(target, scale=0.001)
    sampler_new = cuqi.sampler.PCN(target, scale=0.001)
    assert_true_if_warmup_is_equivalent(sampler_old, sampler_new)

# ============ ULA ============

@pytest.mark.parametrize("target", targets)
def test_ULA_regression_sample(target: cuqi.density.Density):
    """Test the ULA sampler regression."""
    sampler_old = cuqi.legacy.sampler.ULA(target, scale=0.1)
    sampler_new = cuqi.sampler.ULA(target, scale=0.1)
    assert_true_if_sampling_is_equivalent(sampler_old, sampler_new)

@pytest.mark.parametrize("target", targets)
def test_ULA_regression_warmup(target: cuqi.density.Density):
    """Test the ULA sampler regression."""
    sampler_old = cuqi.legacy.sampler.ULA(target, scale=0.001)
    sampler_new = cuqi.sampler.ULA(target, scale=0.001)
    assert_true_if_warmup_is_equivalent(sampler_old, sampler_new)

# ============ MALA ============

@pytest.mark.parametrize("target", targets)
def test_MALA_regression_sample(target: cuqi.density.Density):
    """Test the MALA sampler regression."""
    sampler_old = cuqi.legacy.sampler.MALA(target, scale=1)
    sampler_new = cuqi.sampler.MALA(target, scale=1)
    assert_true_if_sampling_is_equivalent(sampler_old, sampler_new)

@pytest.mark.parametrize("target", targets)
def test_MALA_regression_warmup(target: cuqi.density.Density):
    """Test the MALA sampler regression."""
    sampler_old = cuqi.legacy.sampler.MALA(target, scale=1)
    sampler_new = cuqi.sampler.MALA(target, scale=1)
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
    sampler_old = cuqi.legacy.sampler.LinearRTO(target)
    sampler_new = cuqi.sampler.LinearRTO(target)
    assert_true_if_sampling_is_equivalent(sampler_old, sampler_new)

@pytest.mark.parametrize("target", LinearRTO_targets)
def test_LinearRTO_regression_warmup(target: cuqi.density.Density):
    """Test the LinearRTO sampler regression."""
    sampler_old = cuqi.legacy.sampler.LinearRTO(target)
    sampler_new = cuqi.sampler.LinearRTO(target)
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
    sampler_old = cuqi.legacy.sampler.RegularizedLinearRTO(target, stepsize=1e-3)
    sampler_new = cuqi.sampler.RegularizedLinearRTO(target, stepsize=1e-3)
    assert_true_if_sampling_is_equivalent(sampler_old, sampler_new)

@pytest.mark.parametrize("target", regularized_targets)
def test_RegularizedLinearRTO_regression_warmup(target: cuqi.density.Density):
    """Test the RegularizedLinearRTO sampler regression."""

    sampler_old = cuqi.legacy.sampler.RegularizedLinearRTO(target, stepsize=1e-3)
    sampler_new = cuqi.sampler.RegularizedLinearRTO(target, stepsize=1e-3)
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
    sampler_old = cuqi.legacy.sampler.UGLA(target)
    sampler_new = cuqi.sampler.UGLA(target)
    assert_true_if_sampling_is_equivalent(sampler_old, sampler_new)

@pytest.mark.parametrize("target_dim", [16, 128])
def test_UGLA_regression_warmup(target_dim):
    """Test the UGLA sampler regression."""
    target = create_lmrf_prior_target(dim=target_dim)
    sampler_old = cuqi.legacy.sampler.UGLA(target)
    sampler_new = cuqi.sampler.UGLA(target)
    assert_true_if_warmup_is_equivalent(sampler_old, sampler_new)

# ============== CWMH ============

@pytest.mark.parametrize("target", targets)
def test_CWMH_regression_sample(target: cuqi.density.Density):
    """Test the CWMH sampler regression."""
    sampler_old = cuqi.legacy.sampler.CWMH(target, scale=np.ones(target.dim))
    sampler_new = cuqi.sampler.CWMH(target,
                                                 scale=np.ones(target.dim))
    assert_true_if_sampling_is_equivalent(sampler_old, sampler_new,
                                          Ns=10,
                                          old_idx=[1, -1],
                                          new_idx=[1, -1])

@pytest.mark.parametrize("target", targets)
def test_CWMH_regression_warmup(target: cuqi.density.Density):
    """Test the CWMH sampler regression."""
    sampler_old = cuqi.legacy.sampler.CWMH(target, scale=np.ones(target.dim))
    sampler_new = cuqi.sampler.CWMH(target,
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
    sampler_old = cuqi.legacy.sampler.NUTS(target, adapt_step_size=0.001)
    sampler_new = cuqi.sampler.NUTS(target, step_size=0.001)
    assert_true_if_sampling_is_equivalent(sampler_old, sampler_new, Ns=20)

@pytest.mark.parametrize("target", targets)
def test_NUTS_regression_sample_tune_first_step_only(
    target: cuqi.density.Density):
    """Test the HMC (NUTS) sampler regression."""
    sampler_old = cuqi.legacy.sampler.NUTS(target, adapt_step_size=False)
    sampler_new = cuqi.sampler.NUTS(target, step_size=None)
    assert_true_if_sampling_is_equivalent(sampler_old, sampler_new, Ns=20)

@pytest.mark.parametrize("target", targets)
def test_NUTS_regression_warmup(target: cuqi.density.Density):
    """Test the HMC (NUTS) sampler regression (with warmup)."""
    sampler_old = cuqi.legacy.sampler.NUTS(target, adapt_step_size=True)
    sampler_old._return_burnin = True
    sampler_new = cuqi.sampler.NUTS(target, step_size=None)
    Ns = 20
    Nb = 20
    assert_true_if_warmup_is_equivalent(sampler_old,
                                        sampler_new,
                                        Ns=Ns,
                                        Nb=Nb,
                                        strategy="NUTS")

# ============= MYULA ==============
def create_myula_target(dim=16):
    """Create a target for MYULA."""
    def func(x, restoration_strength=1):
        return x, True
    y = cuqi.testproblem.Deconvolution1D(
        dim=dim).posterior.likelihood 
    x = cuqi.implicitprior.RestorationPrior(
        func, geometry=y.model.domain_geometry)
    # access name property to ensure RestorationPrior
    # name is set 
    x.name
    posterior = cuqi.distribution.Posterior(
        y, x)
    return posterior

def create_myula_smoothed_target(dim=16):
    """Create a target for MYULA."""
    def func(x, restoration_strength=1):
        return x, True
    likelihood = cuqi.testproblem.Deconvolution1D(
        dim=dim).posterior.likelihood 
    restoration_prior = cuqi.implicitprior.RestorationPrior(
        func, geometry=likelihood.model.domain_geometry)
    myprior = cuqi.implicitprior.MoreauYoshidaPrior(prior=restoration_prior, smoothing_strength=0.1)
    posterior = cuqi.distribution.Posterior(
        likelihood, myprior)
    return posterior

def test_myula():
    """ Test creating MYULA sampler."""
    np.random.seed(0)
    posterior = create_myula_target(dim=128)
    np.random.seed(0)
    posterior_smoothed = create_myula_smoothed_target(dim=128)
    np.random.seed(0)
    myula = cuqi.sampler.MYULA(posterior, smoothing_strength=0.1)
    myula.sample(10)
    np.random.seed(0)
    ula = cuqi.sampler.ULA(posterior_smoothed)
    ula.sample(10)
    samples_myula = myula.get_samples()
    samples_ula = ula.get_samples()
    assert samples_ula.Ns == 10
    assert samples_myula.Ns == 10
    assert np.allclose(samples_myula.samples, samples_ula.samples)
    # Assert the name of the MoreauYoshidaPrior created by MYULA is
    # the same name as the corresponding RestorationPrior (both are named x)
    assert myula._smoothed_target.prior.name == 'x'
    assert myula.target.prior.name == 'x'

def test_myula_object_creation_fails_with_target_without_restore_method():
    """ Test that MYULA object creation fails with target that does not
    implement restore method"""
    posterior = cuqi.testproblem.Deconvolution1D(dim=128).posterior
    with pytest.raises(NotImplementedError,
                       match="Using MYULA with a prior that does not have"):
        cuqi.sampler.MYULA(posterior)

def test_myula_object_creation_fails_with_smoothed_target():
    """ Test that MYULA object creation fails with smoothed target."""
    with pytest.raises(ValueError,
                       match="The prior is already smoothed, apply ULA"):
        cuqi.sampler.MYULA(
            create_myula_smoothed_target(dim=128))

# ============= Conjugate ==============

def create_conjugate_target(type:str):
    if type.lower() == 'gaussian-gamma':
        y = cuqi.distribution.Gaussian(0, lambda s: 1/s, name='y')
        s = cuqi.distribution.Gamma(1, 1e-4, name='s')
        return cuqi.distribution.Posterior(y.to_likelihood([0]), s)
    if type.lower() == 'lmrf-gamma':
        x = cuqi.distribution.LMRF(0, lambda s: 1/s, geometry=10, name='x')
        s = cuqi.distribution.Gamma(1, 1e-4, name='s')
        return cuqi.distribution.Posterior(x.to_likelihood(np.zeros(10)), s)
    else:
        raise ValueError(f"Conjugate target type {type} not recognized.")

# ============ Checkpointing ============


# List of all samplers from cuqi.sampler that should be tested for checkpointing + their parameters
checkpoint_targets = [
    cuqi.sampler.ULA(cuqi.testproblem.Deconvolution1D().posterior, scale=0.0001),
    cuqi.sampler.MALA(cuqi.testproblem.Deconvolution1D().posterior, scale=0.0001),
    cuqi.sampler.LinearRTO(cuqi.testproblem.Deconvolution1D().posterior),
    cuqi.sampler.UGLA(create_lmrf_prior_target(dim=16)),
    cuqi.sampler.Direct(cuqi.distribution.Gaussian(np.zeros(10), 1)),
    cuqi.sampler.Conjugate(create_conjugate_target("Gaussian-Gamma")),
    cuqi.sampler.ConjugateApprox(create_conjugate_target("LMRF-Gamma")),
    cuqi.sampler.NUTS(cuqi.testproblem.Deconvolution1D(dim=10).posterior, max_depth=4)
]

# List of samplers from cuqi.sampler that should be skipped for checkpoint testing
skip_checkpoint = [
    cuqi.sampler.Sampler,
    cuqi.sampler.ProposalBasedSampler,
    cuqi.sampler.MH,
    cuqi.sampler.PCN,
    cuqi.sampler.CWMH,
    cuqi.sampler.RegularizedLinearRTO, # Due to the _choose_stepsize method
    cuqi.sampler.HybridGibbs,
    cuqi.sampler.MYULA,
    cuqi.sampler.PnPULA
]

def test_ensure_all_not_skipped_samplers_are_tested_for_checkpointing():
    """Ensure that all samplers from cuqi.sampler, except those skipped, are tested for checkpointing."""

    # List of all samplers from cuqi.sampler that should be tested for checkpointing
    samplers = [
        cls
        for _, cls in inspect.getmembers(cuqi.sampler, inspect.isclass)
        if cls not in skip_checkpoint # use cls here, not name
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
def test_checkpointing(sampler: cuqi.sampler.Sampler):
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
    cuqi.sampler.MH(cuqi.testproblem.Deconvolution1D(dim=10).posterior, scale=0.0001),
    cuqi.sampler.PCN(cuqi.testproblem.Deconvolution1D(dim=10).posterior, scale=0.001),
    cuqi.sampler.CWMH(cuqi.testproblem.Deconvolution1D(dim=10).posterior, scale=0.001),
    cuqi.sampler.ULA(cuqi.testproblem.Deconvolution1D(dim=10).posterior, scale=0.0001),
    cuqi.sampler.MALA(cuqi.testproblem.Deconvolution1D(dim=10).posterior, scale=0.0001),
    cuqi.sampler.LinearRTO(cuqi.testproblem.Deconvolution1D(dim=10).posterior),
    cuqi.sampler.RegularizedLinearRTO(create_regularized_target(dim=16)),
    cuqi.sampler.UGLA(create_lmrf_prior_target(dim=32)),
    cuqi.sampler.NUTS(cuqi.testproblem.Deconvolution1D(dim=10).posterior, max_depth=4),
    cuqi.sampler.MYULA(create_myula_target(dim=128)),
    cuqi.sampler.PnPULA(create_myula_target(dim=128))
]

# List of all classes subclassing samplers.
all_subclassing_sampler_classes= [
    cls
    for _, cls in inspect.getmembers(cuqi.sampler, inspect.isclass)
    if cls not in [cuqi.sampler.Sampler,
                   cuqi.sampler.ProposalBasedSampler,
                   cuqi.sampler.HybridGibbs,
                   cuqi.sampler.Conjugate,
                   cuqi.sampler.ConjugateApprox,
                   cuqi.sampler.Direct]
]
# Make sure that all samplers are tested for state history
@pytest.mark.parametrize("sampler_class", all_subclassing_sampler_classes)
def test_sampler_is_tested_for_state_history(
    sampler_class: cuqi.sampler.Sampler):

    # Find sampler instance that matches the sampler class
    sampler_instance = next(
        (s for s in state_history_targets if isinstance(s, sampler_class)), None)
    if sampler_instance is None:
        raise ValueError(
            f"No sampler instance in the list of state_history_targets matches the sampler class {sampler_class}. "
            "Please add an instance to the list."
        )


@pytest.mark.parametrize("sampler", state_history_targets)
def test_state_keys(sampler: cuqi.sampler.Sampler):
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
def test_history_keys(sampler: cuqi.sampler.Sampler):
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
    cuqi.sampler.ULA: 'scale',
    cuqi.sampler.MALA: 'scale',
    cuqi.sampler.NUTS: 'max_depth',
    cuqi.sampler.MYULA: ['scale'],
    cuqi.sampler.PnPULA: ['scale']
}

@pytest.mark.parametrize("sampler", state_history_targets)
def test_state_is_fully_updated_after_warmup_step(sampler: cuqi.sampler.Sampler):
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
    for _, cls in inspect.getmembers(cuqi.sampler, inspect.isclass)
    if cls not in [cuqi.sampler.Sampler, cuqi.sampler.ProposalBasedSampler, cuqi.sampler.HybridGibbs]
]

# Instances of samplers that should be tested for target=None initialization consistency
initialize_testing_sampler_instances = [
    cuqi.sampler.MH(target=cuqi.testproblem.Deconvolution1D(dim=10).posterior),
    cuqi.sampler.PCN(target=cuqi.testproblem.Deconvolution1D(dim=10).posterior),
    cuqi.sampler.CWMH(target=cuqi.testproblem.Deconvolution1D(dim=10).posterior),
    cuqi.sampler.ULA(target=cuqi.testproblem.Deconvolution1D(dim=10).posterior),
    cuqi.sampler.MALA(target=cuqi.testproblem.Deconvolution1D(dim=10).posterior),
    cuqi.sampler.NUTS(target=cuqi.testproblem.Deconvolution1D(dim=10).posterior),
    cuqi.sampler.LinearRTO(target=cuqi.testproblem.Deconvolution1D(dim=10).posterior),
    cuqi.sampler.RegularizedLinearRTO(target=create_regularized_target(dim=16)),
    cuqi.sampler.UGLA(target=create_lmrf_prior_target(dim=16)),
    cuqi.sampler.Direct(target=cuqi.distribution.Gaussian(np.zeros(10), 1)),
    cuqi.sampler.Conjugate(target=create_conjugate_target("Gaussian-Gamma")),
    cuqi.sampler.ConjugateApprox(target=create_conjugate_target("LMRF-Gamma")),
    cuqi.sampler.MYULA(target=create_myula_target(dim=16)),
    cuqi.sampler.PnPULA(target=create_myula_target(dim=16))
]


@pytest.mark.parametrize("sampler_class", initialize_testing_sampler_classes)
def test_target_None_init_in_samplers(sampler_class):
    """ Test all samplers can be initialized with target=None. """
    sampler = sampler_class()
    assert sampler.target is None, f"Sampler {sampler_class} failed to initialize with target=None"

@pytest.mark.parametrize("sampler_class", initialize_testing_sampler_classes)
def test_sampler_initialization_consistency(sampler_class: cuqi.sampler.Sampler):
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
            assert (np.isnan(attr1) and np.isnan(attr2)) \
                or np.allclose(attr1, attr2), f"Number: Attribute '{key}' differs after initialization"
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

# ============ Testing of Conjugate handling ============

def test_conjugate_invalid_target_type():
    """ Test that the Conjugate sampler requires a target of type Posterior. """
    sampler = cuqi.sampler.Conjugate()
    invalid_target = cuqi.distribution.Gaussian(0, 1) # Not a Posterior
    with pytest.raises(TypeError, match="Conjugate sampler requires a target of type Posterior"):
        sampler.target = invalid_target

def test_conjugate_invalid_pair():
    """ Test that useful error message is raised when conjugate pair is not supported. """
    prior = cuqi.distribution.Gaussian(0, 1, name="x")
    likelihood = cuqi.distribution.Gamma(lambda x: x, 1, name="y").to_likelihood([0])
    posterior = cuqi.distribution.Posterior(likelihood, prior)

    with pytest.raises(ValueError, match="Conjugacy is not defined for likelihood"):
        cuqi.sampler.Conjugate(target=posterior)

def test_conjugate_wrong_name_for_conjugate_parameter():
    """ Test that useful error message is raised when name of conjugate parameter is wrong. """
    posterior = create_conjugate_target("Gaussian-Gamma")
    # Modify likelihood to use wrong name for conjugate parameter
    posterior.likelihood.distribution.cov = lambda d: 1/d

    with pytest.raises(ValueError, match="Unable to find conjugate parameter"):
        cuqi.sampler.Conjugate(target=posterior)

def test_conjugate_wrong_var_for_conjugate_parameter():
    """ Test that useful error message is raised when conjugate parameter is defined on wrong mutable variable. """
    y = cuqi.distribution.Gaussian(0, sqrtprec=lambda s: 1/s, name='y')
    s = cuqi.distribution.Gamma(1, 1e-4, name='s')
    posterior =  cuqi.distribution.Posterior(y.to_likelihood([0]), s)

    with pytest.raises(ValueError, match="RegularizedGaussian-ModifiedHalfNormal conjugacy does not support the conjugate parameter s in the sqrtprec attribute. Only cov and prec"):
        cuqi.sampler.Conjugate(target=posterior)

def test_conjugate_wrong_equation_for_conjugate_parameter():
    """ Test that useful error message is raised when equation for conjugate parameter is not supported. """
    posterior = create_conjugate_target("Gaussian-Gamma")
    # Modify likelihood to not invert parameter in covariance
    posterior.likelihood.distribution.cov = lambda s: s

    with pytest.raises(ValueError, match="Gaussian-Gamma conjugate pair defined via covariance requires cov: lambda x : s/x for the conjugate parameter"):
        cuqi.sampler.Conjugate(target=posterior)

def create_invalid_conjugate_target(target_type: str, param_name: str, invalid_func):
    """ Create a target with invalid conjugate parameter equations. """

    if target_type.lower() == 'gaussian-gamma':
        if param_name == "cov":
            y = cuqi.distribution.Gaussian(0, invalid_func, name='y')
        elif param_name == "prec":
            y = cuqi.distribution.Gaussian(0, prec=invalid_func, name='y')
        s = cuqi.distribution.Gamma(1, 1e-4, name='s')
        return cuqi.distribution.Posterior(y.to_likelihood([0]), s)
    
    elif target_type.lower() == 'regularizedgaussian-gamma':
        if param_name == "cov":
            x = cuqi.implicitprior.RegularizedGaussian(0, invalid_func, constraint="nonnegativity", name='x')
        elif param_name == "prec":
            x = cuqi.implicitprior.RegularizedGaussian(0, prec=invalid_func, constraint="nonnegativity", name='x')
        s = cuqi.distribution.Gamma(1, 1e-4, name='s')
        return cuqi.distribution.Posterior(x.to_likelihood([0]), s)
    
    elif target_type.lower() == 'lmrf-gamma':
        if param_name == "scale":
            x = cuqi.distribution.LMRF(0, scale=invalid_func, geometry=10, name='x')
        s = cuqi.distribution.Gamma(1, 1e-4, name='s')
        return cuqi.distribution.Posterior(x.to_likelihood(np.zeros(10)), s)

    elif target_type.lower() == 'gmrf-gamma':
        if param_name == "prec":
            x = cuqi.distribution.GMRF(0, prec=invalid_func, geometry=10, name='x')
        s = cuqi.distribution.Gamma(1, 1e-4, name='s')
        return cuqi.distribution.Posterior(x.to_likelihood(np.zeros(10)), s)
    
    elif target_type.lower() == 'regularizedgmrf-gamma':
        if param_name == "prec":
            x = cuqi.implicitprior.RegularizedGMRF(np.zeros(10), prec=invalid_func, constraint="nonnegativity", name='x')
        s = cuqi.distribution.Gamma(1, 1e-4, name='s')
        return cuqi.distribution.Posterior(x.to_likelihood(np.zeros(10)), s)
    
    else:
        raise ValueError(f"Conjugate target type {target_type} not recognized.")

@pytest.mark.parametrize("target_type, param_name, invalid_func, expected_error", [
    ("gaussian-gamma", "cov", lambda s: s, "Gaussian-Gamma conjugate pair defined via covariance requires cov: lambda x : s/x for the conjugate parameter"),
    ("regularizedgaussian-gamma", "cov", lambda s: s, "Regularized Gaussian-Gamma conjugacy defined via covariance requires cov: lambda x : s/x for the conjugate parameter"),
    ("lmrf-gamma", "scale", lambda s: s, "Approximate conjugate sampler only works with Gamma prior on the inverse of the scale parameter of the LMRF likelihood"),
])
def test_conjugate_wrong_equation_for_conjugate_parameter_supported_cases(target_type, param_name, invalid_func, expected_error):
    """ Test that useful error message is raised when conjugate parameter has the wrong equation. """
    posterior = create_invalid_conjugate_target(target_type, param_name, invalid_func)
    
    with pytest.raises(ValueError, match=expected_error):
        if target_type == "lmrf-gamma":
            cuqi.sampler.ConjugateApprox(target=posterior)
        else:
            cuqi.sampler.Conjugate(target=posterior)

# ============ Testing of HybridGibbs ============

def test_HybridGibbs_initial_point_setting():
    """ Test that the HybridGibbs sampler adheres to the initial point set by sampling strategy. """

    # Forward model
    A, y_data, _ = cuqi.testproblem.Deconvolution1D(dim=10).get_components()

    # Bayesian Problem
    d = cuqi.distribution.Uniform(0, 100)
    s = cuqi.distribution.Uniform(0, 100)
    x = cuqi.distribution.Gaussian(0, lambda d: 1/d, geometry=A.domain_geometry)
    y = cuqi.distribution.Gaussian(A@x, lambda s: 1/s)

    # Joint distribution
    joint = cuqi.distribution.JointDistribution(x, y, d, s)

    # Posterior
    posterior = joint(y=y_data)

    # Sampling strategy
    sampling_strategy = {
        "d" : cuqi.sampler.MH(initial_point=3),
        "s" : cuqi.sampler.MH(),
        "x" : cuqi.sampler.MALA(initial_point=0.5*np.ones(10))
    }

    # Hybrid Gibbs sampler
    sampler = cuqi.sampler.HybridGibbs(posterior, sampling_strategy=sampling_strategy)

    # Test that the initial point is set correctly in Gibbs
    assert sampler.current_samples["d"] == 3
    assert sampler.current_samples["s"] == 1 # Default initial point of MH is 1.
    assert np.allclose(sampler.current_samples["x"], 0.5*np.ones(10))

def test_HybridGibbs_stores_acc_rate():
    """ Test that the HybridGibbs sampler stores the acceptance rate of the samplers correctly. Also ensures that history is maintained during warmup and sampling. """

    # Number of warmup and sampling steps
    Ns = 25
    Nb = 25
    
    # Bayesian Problem
    s = cuqi.distribution.Gaussian(1, 1)
    d = cuqi.distribution.Uniform(1, 100)
    x = cuqi.distribution.Gaussian(lambda s: s, lambda d: 1/d, geometry=1)

    # Joint distribution
    joint = cuqi.distribution.JointDistribution(x, d, s)

    # Sampling strategy
    sampling_strategy = {
        "d" : cuqi.sampler.MH(initial_point=3),
        "s" : cuqi.sampler.PCN(initial_point=3),
        "x" : cuqi.sampler.MALA(initial_point=0)
    }

    # Hybrid Gibbs sampler
    sampler = cuqi.sampler.HybridGibbs(joint, sampling_strategy=sampling_strategy)

    # Run the sampler for a few steps
    sampler.warmup(Nb).sample(Ns)

    # Check that the samplers have acc rate correctly updated and maintained
    assert len(sampler.samplers["d"].get_history()["history"]["_acc"]) == Nb + Ns + 1
    assert len(sampler.samplers["s"].get_history()["history"]["_acc"]) == Nb + Ns + 1
    assert len(sampler.samplers["x"].get_history()["history"]["_acc"]) == Nb + Ns + 1

def test_HybridGibbs_updates_state_only_after_accepting_sample():
    """ Test that the HybridGibbs sampler updates the state only after accepting a sample. """

    # Number of warmup and sampling steps
    Ns = 25
    Nb = 25
    
    # Bayesian Problem
    s = cuqi.distribution.Gaussian(1, 1)
    d = cuqi.distribution.Uniform(1, 100)
    x = cuqi.distribution.Gaussian(lambda s: s, lambda d: 1/d, geometry=1)

    # Joint distribution
    joint = cuqi.distribution.JointDistribution(x, d, s)

    # Sampling strategy
    sampling_strategy = {
        "d" : cuqi.sampler.MH(initial_point=3),
        "s" : cuqi.sampler.PCN(initial_point=3),
        "x" : cuqi.sampler.MALA(initial_point=0, scale=1e-1) # Relatively high scale may lead to no accepted samples
    }

    # Hybrid Gibbs sampler
    sampler = cuqi.sampler.HybridGibbs(joint, sampling_strategy=sampling_strategy)

    # Run the sampler for a few steps
    sampler.warmup(Nb).sample(Ns)

    # Store states of samplers
    sampler_states = {key: sampler.samplers[key].get_state() for key in sampler.samplers.keys()}

    # Run warmup and sampling again
    sampler.warmup(Nb).sample(Ns)

    # Verify that the state is only updated if a sample is accepted.
    for key in sampler.samplers.keys():

        # Get new state for specific sampler
        new_state = sampler.samplers[key].get_state()

        # In case sampler has no accepted samples, the state should not have been updated
        if (np.sum(sampler.samplers[key].get_history()["history"]["_acc"][Ns+Nb+1:]) == 0):
            assert sampler_states[key] == new_state, f"Sampler {key} state was erroneously updated in Gibbs scheme, but no accepted samples. State: \n {new_state}"
        # In case sampler has accepted samples, the state should have been updated
        else:
            assert sampler_states[key] != new_state, f"Sampler {key} state was erroneously not updated in Gibbs scheme, even when new samples were accepted. State: \n {new_state}"

def HybridGibbs_target_1():
    """ Create a target for the HybridGibbs sampler. """
    # Forward problem
    np.random.seed(0)
    A, y_data, info = cuqi.testproblem.Deconvolution1D(
        dim=128, phantom='sinc', noise_std=0.001).get_components()
    
    # Bayesian Inverse Problem
    s = cuqi.distribution.Gamma(1, 1e-4)
    x = cuqi.distribution.GMRF(np.zeros(A.domain_dim), 50)
    y = cuqi.distribution.Gaussian(A@x, lambda s: 1/s)
    
    # Posterior
    target = cuqi.distribution.JointDistribution(y, x, s)(y=y_data)

    return target


def test_NUTS_within_HybridGibbs_regression_sample_and_warmup(copy_reference):
    """ Test that using NUTS sampler within HybridGibbs sampler works as
    expected."""

    Nb=10
    Ns=10

    target = HybridGibbs_target_1()

    sampling_strategy = {
        "x" : cuqi.sampler.NUTS(max_depth=7),
        "s" : cuqi.sampler.Conjugate()
    }

    # Here we do 1 internal steps with NUTS for each Gibbs step
    num_sampling_steps = {
        "x" : 2,
        "s" : 1
    }

    sampler = cuqi.sampler.HybridGibbs(
        target, sampling_strategy, num_sampling_steps)
    
    np.random.seed(0)
    sampler.warmup(Nb)
    sampler.sample(Ns)
    samples = sampler.get_samples()

    # Read samples from reference
    file = copy_reference("data/s_x_NUTS_within_HybridGibbs.npz")
    reference = np.load(file)
    reference_s = reference["s"]
    reference_x = reference["x"]

    # Compare samples
    assert np.allclose(samples["s"].samples, reference_s, rtol=1e-3)
    assert np.allclose(samples["x"].samples, reference_x, rtol=1e-3)


# ============ Test for sampling with bounded distributions ============
sampler_instances_for_bounded_distribution = [
    cuqi.sampler.MH(
        target=cuqi.distribution.Beta(0.5, 0.5),
        initial_point=np.array([0.1]),
        scale=0.1,
    ),
    cuqi.sampler.MALA(
        cuqi.distribution.Beta(0.5, 0.5),
        initial_point=np.array([0.1]),
        scale=0.1,
    ),
    cuqi.sampler.NUTS(
        cuqi.distribution.Beta(0.5, 0.5), initial_point=np.array([0.1])
    ),
    cuqi.sampler.MH(
        target=cuqi.distribution.Beta(np.array([0.5, 0.5]), np.array([0.5, 0.5])),
        initial_point=np.array([0.1, 0.1]),
        scale=0.1,
    ),
    cuqi.sampler.CWMH(
        target=cuqi.distribution.Beta(
            np.array([0.5, 0.5]), np.array([0.5, 0.5])
        ),
        initial_point=np.array([0.1, 0.1]),
        scale=0.1,
    ),
    cuqi.sampler.MALA(
        cuqi.distribution.Beta(np.array([0.5, 0.5]), np.array([0.5, 0.5])),
        initial_point=np.array([0.1, 0.1]),
        scale=0.1,
    ),
    cuqi.sampler.NUTS(
        cuqi.distribution.Beta(np.array([0.5, 0.5]), np.array([0.5, 0.5])),
        initial_point=np.array([0.1, 0.1]),
    ),
]

@pytest.mark.parametrize("sampler", sampler_instances_for_bounded_distribution)
def test_if_invalid_sample_accepted(sampler: cuqi.sampler.Sampler):
    sampler.sample(50)
    samples = sampler.get_samples().samples
    tol = 1e-8
    assert (
        samples.min() > 0.0 - tol and samples.max() < 1.0 + tol
    ), f"Invalid samples accepted for sampler {sampler.__class__.__name__}."


# Test NUTS acceptance rate
@pytest.mark.parametrize(
    "sampler",
    [
        cuqi.sampler.NUTS(cuqi.distribution.Gaussian(0, 1)),
        cuqi.sampler.NUTS(cuqi.distribution.DistributionGallery('donut'))
    ],
)
def test_nuts_acceptance_rate(sampler: cuqi.sampler.Sampler):
    """ Test that the NUTS sampler correctly updates the acceptance rate. """
    # Fix random seed for reproducibility, but the test should be robust to seed
    np.random.seed(0)

    # Sample:
    sampler.warmup(100).sample(100)

    # Compute number of times samples were updated:
    samples = sampler.get_samples().samples
    counter = 0
    for i in range(1, samples.shape[1]):
        if np.any(samples[:, i] != samples[:, i - 1]):
            counter += 1

    # Compute the number of accepted samples according to the sampler
    acc_rate_sum = sum(sampler._acc[2:])

    assert np.isclose(counter, acc_rate_sum), "NUTS sampler does not update acceptance rate correctly: "+str(counter)+" != "+str(acc_rate_sum)

# ============ Testing of AffineModel with RTO-type samplers ============

def test_LinearRTO_with_AffineModel_is_equivalent_to_LinearModel_and_shifted_data():
    """ Test that LinearRTO with AffineModel is equivalent to LinearRTO with LinearModel and shifted data. """

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
    samples_linear = sampler_linear.warmup(2).sample(10).get_samples()

    # Define AffineModel
    affine_model = cuqi.model.AffineModel(A, shift)

    # Set up LinearRTO with AffineModel
    y = cuqi.distribution.Gaussian(affine_model, 0.01**2)
    posterior_affine = cuqi.distribution.JointDistribution(x, y)(y=y_obs)

    # Set up LinearRTO with AffineModel
    sampler_affine = cuqi.sampler.LinearRTO(posterior_affine)

    # Sample with fixes seed
    np.random.seed(0)
    samples_affine = sampler_affine.warmup(2).sample(10).get_samples()

    # Check that the samples are the same
    assert np.allclose(samples_linear.samples, samples_affine.samples)

def test_RegularizedLinearRTO_with_AffineModel_is_equivalent_to_RegularizedLinearModel_and_shifted_data():
    """ Test that RegularizedLinearRTO with AffineModel is equivalent to RegularizedLinearRTO with RegularizedLinearModel and shifted data. """

    # Define LinearModel and data
    A, y_obs, _ = cuqi.testproblem.Deconvolution1D().get_components()

    # Define Shift
    shift = np.random.rand(A.domain_dim)

    # Define Bayesian Problem
    x = cuqi.implicitprior.NonnegativeGMRF(np.zeros(A.domain_dim), 100)
    y = cuqi.distribution.Gaussian(A@x, 0.01**2)
    posterior = cuqi.distribution.JointDistribution(x, y)(y=y_obs-shift)

    # Set up LinearRTO with both models
    sampler_linear = cuqi.sampler.RegularizedLinearRTO(posterior)

    # Sample with fixes seed
    np.random.seed(0)
    samples_linear = sampler_linear.warmup(2).sample(10).get_samples()

    # Define AffineModel
    affine_model = cuqi.model.AffineModel(A, shift)

    # Set up LinearRTO with AffineModel
    y = cuqi.distribution.Gaussian(affine_model, 0.01**2)
    posterior_affine = cuqi.distribution.JointDistribution(x, y)(y=y_obs)

    # Set up LinearRTO with AffineModel
    sampler_affine = cuqi.sampler.RegularizedLinearRTO(posterior_affine)

    # Sample with fixes seed
    np.random.seed(0)
    samples_affine = sampler_affine.warmup(2).sample(10).get_samples()

    # Check that the samples are the same
    assert np.allclose(samples_linear.samples, samples_affine.samples)

def test_UGLA_with_AffineModel_is_equivalent_to_LinearModel_and_shifted_data():
    """ Test that UGLA with AffineModel is equivalent to LinearModel and shifted data. """
    
    # Define LinearModel and data
    A, y_obs, _ = cuqi.testproblem.Deconvolution1D().get_components()

    # Define Shift
    shift = np.random.rand(A.domain_dim)

    # Define Bayesian Problem
    x = cuqi.distribution.LMRF(np.zeros(A.domain_dim), 0.01)
    y = cuqi.distribution.Gaussian(A@x, 0.01**2)
    posterior = cuqi.distribution.JointDistribution(x, y)(y=y_obs-shift)

    # Set up LinearRTO with both models
    sampler_linear = cuqi.sampler.UGLA(posterior)

    # Sample with fixes seed
    np.random.seed(0)
    samples_linear = sampler_linear.warmup(2).sample(10).get_samples()

    # Define AffineModel
    affine_model = cuqi.model.AffineModel(A, shift)

    # Set up LinearRTO with AffineModel
    y = cuqi.distribution.Gaussian(affine_model, 0.01**2)
    posterior_affine = cuqi.distribution.JointDistribution(x, y)(y=y_obs)

    # Set up LinearRTO with AffineModel
    sampler_affine = cuqi.sampler.UGLA(posterior_affine)

    # Sample with fixes seed
    np.random.seed(0)
    samples_affine = sampler_affine.warmup(2).sample(10).get_samples()

    # Check that the samples are the same
    assert np.allclose(samples_linear.samples, samples_affine.samples)

# ============ Test for sampling with RandomVariable prior against Distribution prior ============
samplers_for_rv_against_dist = [cuqi.sampler.MALA, 
                                cuqi.sampler.ULA,
                                cuqi.sampler.MH,
                                cuqi.sampler.PCN,
                                cuqi.sampler.CWMH,
                                cuqi.sampler.NUTS,
                                cuqi.sampler.LinearRTO]

@pytest.mark.parametrize("sampler", samplers_for_rv_against_dist)
def test_RandomVariable_prior_against_Distribution_prior(sampler: cuqi.sampler.Sampler):
    """ Test RandomVariable prior is equivalent to Distribution prior for 
        MALA, ULA, MH, PCN, CWMH, NUTS and LinearRTO. 
    """

    # Set dim
    dim = 32

    # Extract model and data
    A, y_data, info = cuqi.testproblem.Deconvolution1D(dim=32, phantom='square').get_components()

    # Set up RandomVariable prior and do posterior sampling
    np.random.seed(0)
    x_rv = cuqi.distribution.Gaussian(0.5*np.ones(dim), 0.1).rv
    y_rv = cuqi.distribution.Gaussian(A@x_rv, 0.001).rv
    joint_rv = cuqi.distribution.JointDistribution(x_rv, y_rv)(y_rv=y_data)
    sampler_rv = sampler(joint_rv)
    sampler_rv.sample(10)
    samples_rv = sampler_rv.get_samples()
    
    # Set up Distribution prior and do posterior sampling
    np.random.seed(0)
    x_dist = cuqi.distribution.Gaussian(0.5*np.ones(dim), 0.1)
    y_dist = cuqi.distribution.Gaussian(A@x_dist, 0.001)
    joint_dist = cuqi.distribution.JointDistribution(x_dist, y_dist)(y_dist=y_data)
    sampler_dist = sampler(joint_dist)
    sampler_dist.sample(10)
    samples_dist = sampler_dist.get_samples()

    assert np.allclose(samples_rv.samples, samples_dist.samples)

def test_RandomVariable_prior_against_Distribution_prior_regularized_RTO():
    """ Test RandomVariable prior is equivalent to Distribution prior for 
        RegularizedLinearRTO. 
    """

    # Set dim
    dim = 32

    # Extract model and data
    A, y_data, info = cuqi.testproblem.Deconvolution1D(dim=32, phantom='square').get_components()

    # Set up RandomVariable prior and do posterior sampling
    np.random.seed(0)
    x_rv = cuqi.implicitprior.RegularizedGaussian(0.5*np.ones(dim), 0.1, constraint="nonnegativity").rv
    y_rv = cuqi.distribution.Gaussian(A@x_rv, 0.001).rv
    joint_rv = cuqi.distribution.JointDistribution(x_rv, y_rv)(y_rv=y_data)
    sampler_rv = cuqi.sampler.RegularizedLinearRTO(joint_rv)
    sampler_rv.sample(10)
    samples_rv = sampler_rv.get_samples()
    
    # Set up Distribution prior and do posterior sampling
    np.random.seed(0)
    x_dist = cuqi.implicitprior.RegularizedGaussian(0.5*np.ones(dim), 0.1, constraint="nonnegativity")
    y_dist = cuqi.distribution.Gaussian(A@x_dist, 0.001)
    joint_dist = cuqi.distribution.JointDistribution(x_dist, y_dist)(y_dist=y_data)
    sampler_dist = cuqi.sampler.RegularizedLinearRTO(joint_dist)
    sampler_dist.sample(10)
    samples_dist = sampler_dist.get_samples()

    assert np.allclose(samples_rv.samples, samples_dist.samples)

def test_RandomVariable_prior_against_Distribution_prior_UGLA_Conjugate_ConjugateApprox_HybridGibbs():
    """ Test RandomVariable prior is equivalent to Distribution prior for 
        UGLA, Conjugate, ConjugateApprox and HybridGibbs samplers.
    """

    # Forward problem
    A, y_data, info = cuqi.testproblem.Deconvolution1D(dim=28, phantom='square', noise_std=0.001).get_components()

    # Random seed
    np.random.seed(0)

    # Bayesian Inverse Problem
    d = cuqi.distribution.Gamma(1, 1e-4)
    s = cuqi.distribution.Gamma(1, 1e-4)
    x = cuqi.distribution.LMRF(0, lambda d: 1/d, geometry=A.domain_geometry)
    y = cuqi.distribution.Gaussian(A@x, lambda s: 1/s)

    # Posterior
    target = cuqi.distribution.JointDistribution(y, x, s, d)(y=y_data)

    # Sampling strategy
    sampling_strategy = {
        "x" : cuqi.sampler.UGLA(),
        "s" : cuqi.sampler.Conjugate(),
        "d" : cuqi.sampler.ConjugateApprox()
    }

    # Gibbs sampler
    sampler = cuqi.sampler.HybridGibbs(target, sampling_strategy)

    # Run sampler
    sampler.warmup(50)
    sampler.sample(200)
    samples = sampler.get_samples()

    # Random seed
    np.random.seed(0)

    # Bayesian Inverse Problem
    d_rv = cuqi.distribution.Gamma(1, 1e-4).rv
    s_rv = cuqi.distribution.Gamma(1, 1e-4).rv
    x_rv = cuqi.distribution.LMRF(0, lambda d_rv: 1/d_rv, geometry=A.domain_geometry).rv
    y_rv = cuqi.distribution.Gaussian(A@x_rv, lambda s_rv: 1/s_rv).rv

    # Posterior
    target_rv = cuqi.distribution.JointDistribution(y_rv, x_rv, s_rv, d_rv)(y_rv=y_data)

    # Sampling strategy
    sampling_strategy_rv = {
        "x_rv" : cuqi.sampler.UGLA(),
        "s_rv" : cuqi.sampler.Conjugate(),
        "d_rv" : cuqi.sampler.ConjugateApprox()
    }

    # Gibbs sampler
    sampler_rv = cuqi.sampler.HybridGibbs(target_rv, sampling_strategy_rv)

    # Run sampler
    sampler_rv.warmup(50)
    sampler_rv.sample(200)
    samples_rv = sampler_rv.get_samples()

    assert np.allclose(samples['x'].samples, samples_rv['x_rv'].samples)

def Conjugate_GaussianGammaPair():
    """ Unit test whether Conjugacy Pair (Gaussian, Gamma) constructs the right distribution """
    x = cuqi.distribution.Gamma(1.0, 2.0)
    y = cuqi.distribution.Gaussian(np.array([1.0, 1.0]), prec = lambda x : x)
    joint = cuqi.distributionJointDistribution(x, y)(y = np.array([2, 1]))
    sampler = cuqi.sampler.Conjugate(joint)
    conj = sampler.conjugate_distribution()

    assert isinstance(conj, type(x))
    assert conj.shape == 2.0
    assert conj.scale == 0.4

def Conjugate_RegularizedGaussianGammaPair():
    """ Unit test whether Conjugacy Pair (RegularizedGaussian, Gamma) constructs the right distribution """
    x = cuqi.distribution.Gamma(1.0, 2.0)
    y = cuqi.implicitprior.RegularizedGaussian(np.array([1.0, 1.0]), prec = lambda x : x, constraint="nonnegativity")
    joint = cuqi.distribution.JointDistribution(x, y)(y = np.array([1, 0]))
    sampler = cuqi.sampler.Conjugate(joint)
    conj = sampler.conjugate_distribution()

    assert isinstance(conj, type(x))
    assert conj.shape == 1.5
    assert conj.scale == 0.4

def Conjugate_RegularizedUnboundedUniformGammaPair():
    """ Unit test whether Conjugacy Pair (RegularizedUnboundedUniform, Gamma) constructs the right distribution """
    x = cuqi.distribution.Gamma(1.0, 2.0)
    y = cuqi.implicitprior.RegularizedUnboundedUniform(regularization='tv', strength = lambda x : x, geometry = cuqi.geometry.Continuous1D(2))
    joint = cuqi.distribution.JointDistribution(x, y)(y = np.array([2, 0]))
    sampler = cuqi.sampler.Conjugate(joint)
    conj = sampler.conjugate_distribution()

    assert isinstance(conj, type(x))
    assert conj.shape == 2.0
    assert conj.scale == 0.25

def Conjugate_RegularizedGaussianModifiedHalfNormalPair():
    """ Unit test whether Conjugacy Pair (RegularizedGaussian, ModifiedHalfNormal) constructs the right distribution """
    x = cuqi.distribution.ModifiedHalfNormal(1.0, 3.0, -3.0)
    y = cuqi.implicitprior.RegularizedGaussian(np.array([1.0, 1.0]), prec = lambda x : x**2, regularization='tv', strength = lambda x : x, geometry = cuqi.geometry.Continuous1D(2))
    joint = cuqi.distribution.JointDistribution(x, y)(y = np.array([2, 0]))
    sampler = cuqi.sampler.Conjugate(joint)
    conj = sampler.conjugate_distribution()

    assert isinstance(conj, type(x))
    assert conj.alpha == 3.0
    assert conj.beta == 4.0
    assert conj.gamma == -5.0


def test_RegularizedGaussianHierchical_sample_regression():
    np.random.seed(24601)
    n = 2

    A_mat = np.array([[1,2],[3,4]])
    y_data = np.array([1,2])

    A = cuqi.model.LinearModel(A_mat, domain_geometry=cuqi.geometry.Continuous1D(2), range_geometry=cuqi.geometry.Continuous1D(2))

    l = cuqi.distribution.Gamma(1, 1e-4)
    d = cuqi.distribution.ModifiedHalfNormal(1, 1e-4, -1e-4)
    x = cuqi.implicitprior.RegularizedGMRF(mean = np.zeros(n), prec = lambda d : 0.1*d**2, regularization = "TV", strength = lambda d : 50*d, geometry = A.domain_geometry)
    y =cuqi.distribution. Gaussian(A@x, prec = lambda l : l)

    joint = cuqi.distribution.JointDistribution(x, y, d, l)
    posterior = joint(y=y_data)

    sampling_strategy = {
                'x': cuqi.sampler.RegularizedLinearRTO(maxit=50, penalty_parameter=10, adaptive = False),
                'd': cuqi.sampler.Conjugate(),
                'l': cuqi.sampler.Conjugate(),
                }
    sampler = cuqi.sampler.HybridGibbs(posterior, sampling_strategy)

    sampler.warmup(10)
    sampler.sample(10)

    samples = sampler.get_samples().burnthin(10)

    assert np.allclose(samples['x'].samples, np.array([[0.28756343, 0.31439747, 0.29883007, 0.29241259, 0.3142019, 0.28276159, 0.27819533, 0.32370642, 0.24194951, 0.30077088],
                                                        [0.28756339, 0.31439747, 0.29883007, 0.29409678, 0.3142019, 0.28276159, 0.27819533, 0.32371729, 0.24194986, 0.30077088]]))

    assert np.allclose(samples['l'].samples, np.array([[43.51120066, 109.27863688, 117.44177758, 93.02865816, 2.09937242,22.28328818, 58.69566463, 66.46108287, 21.68571243, 76.04025099]]))
    
    assert np.allclose(samples['d'].samples, np.array([[9.25399315, 5.04438304, 26.84002718, 7.69622219, 8.47935032, 5.15752285, 16.4884862, 13.44909853, 3.34200395, 5.71966806]]))

def test_RegularizedLinearRTO_ScipyLinearLSQ_option_valid():
    n = 2

    A_mat = np.array([[1,2],[3,4]])
    y_data = np.array([1,2])

    A = cuqi.model.LinearModel(A_mat, domain_geometry=cuqi.geometry.Continuous1D(2), range_geometry=cuqi.geometry.Continuous1D(2))

    x = cuqi.implicitprior.RegularizedGMRF(mean = np.zeros(n), prec = 1, constraint = "nonnegativity")
    y = cuqi.distribution. Gaussian(A@x, prec = 1)
    
    joint = cuqi.distribution.JointDistribution(x, y)
    posterior = joint(y=y_data)

    sampler = cuqi.sampler.RegularizedLinearRTO(posterior, solver = "ScipyLinearLSQ")
    assert sampler.solver == "ScipyLinearLSQ"

def test_RegularizedLinearRTO_ScipyLinearLSQ_option_invalid():
    n = 2

    A_mat = np.array([[1,2],[3,4]])
    y_data = np.array([1,2])

    A = cuqi.model.LinearModel(A_mat, domain_geometry=cuqi.geometry.Continuous1D(2), range_geometry=cuqi.geometry.Continuous1D(2))

    x = cuqi.implicitprior.RegularizedGMRF(mean = np.zeros(n), prec = 0, regularization= "TV", strength = 10, geometry = A.domain_geometry)
    y = cuqi.distribution. Gaussian(A@x, prec = 1)
    
    joint = cuqi.distribution.JointDistribution(x, y)
    posterior = joint(y=y_data)

    with pytest.raises(ValueError, match="ScipyLinearLSQ"):
        sampler = cuqi.sampler.RegularizedLinearRTO(posterior, solver = "ScipyLinearLSQ")

def test_RegularizedLinearRTO_inner_initial_point_setting():
    # Define LinearModel and data
    A, y_obs, _ = cuqi.testproblem.Deconvolution1D().get_components()

    # Define Bayesian Problem
    x = cuqi.implicitprior.NonnegativeGMRF(np.zeros(A.domain_dim), 100)
    y = cuqi.distribution.Gaussian(A@x, 0.01**2)
    posterior = cuqi.distribution.JointDistribution(x, y)(y=y_obs)

    # Set up RegularizedLinearRTO with three solvers
    sampler1 = cuqi.sampler.RegularizedLinearRTO(posterior, maxit=10, inner_initial_point="previous_sample", tol=1e-8)
    sampler2 = cuqi.sampler.RegularizedLinearRTO(posterior, maxit=10, inner_initial_point="MAP", tol=1e-8)
    sampler3 = cuqi.sampler.RegularizedLinearRTO(posterior, maxit=10, inner_initial_point=np.ones(A.domain_dim), tol=1e-8)

    # Sample with fixed seed
    np.random.seed(0)
    sampler1.sample(5)
    np.random.seed(0)
    sampler2.sample(5)
    np.random.seed(0)
    sampler3.sample(5)

    assert np.allclose(sampler1.inner_initial_point, sampler1.current_point, rtol=1e-5)
    assert np.allclose(sampler2.inner_initial_point, sampler2._map, rtol=1e-5)
    assert np.allclose(sampler3.inner_initial_point, np.ones(A.domain_dim), rtol=1e-5)

def test_RegularizedLinearRTO_ScipyLinearLSQ_against_ScipyMinimizer_and_against_FISTA():
    # Define LinearModel and data
    A, y_obs, _ = cuqi.testproblem.Deconvolution1D().get_components()

    # Define Bayesian Problem
    x = cuqi.implicitprior.NonnegativeGMRF(np.zeros(A.domain_dim), 100)
    y = cuqi.distribution.Gaussian(A@x, 0.01**2)
    posterior = cuqi.distribution.JointDistribution(x, y)(y=y_obs)

    # Set up RegularizedLinearRTO with three solvers
    sampler1 = cuqi.sampler.RegularizedLinearRTO(posterior, solver="ScipyMinimizer", maxit=1000, tol=1e-8)
    sampler2 = cuqi.sampler.RegularizedLinearRTO(posterior, solver="ScipyLinearLSQ", maxit=1000, tol=1e-8)
    sampler3 = cuqi.sampler.RegularizedLinearRTO(posterior, solver="FISTA", maxit=1000, tol=1e-8)

    # Sample with fixed seed
    np.random.seed(0)
    samples1 = sampler1.sample(5).get_samples()
    np.random.seed(0)
    samples2 = sampler2.sample(5).get_samples()
    np.random.seed(0)
    samples3 = sampler3.sample(5).get_samples()

    assert np.allclose(samples1.samples.mean(), samples2.samples.mean(), rtol=1e-5)
    assert np.allclose(samples1.samples.mean(), samples3.samples.mean(), rtol=1e-5)

# ============ Start testing sampler callback ============
# Samplers that should be tested for callback
callback_testing_sampler_classes = [
    cls
    for _, cls in inspect.getmembers(cuqi.sampler, inspect.isclass)
    if cls not in [cuqi.sampler.Sampler, cuqi.sampler.ProposalBasedSampler]
]

# Instances of samplers that should be tested for callback
callback_testing_sampler_instances = [
    cuqi.sampler.MH(
        target=cuqi.testproblem.Deconvolution1D(dim=10).posterior
    ),
    cuqi.sampler.PCN(
        target=cuqi.testproblem.Deconvolution1D(dim=10).posterior
    ),
    cuqi.sampler.CWMH(
        target=cuqi.testproblem.Deconvolution1D(dim=10).posterior
    ),
    cuqi.sampler.ULA(
        target=cuqi.testproblem.Deconvolution1D(dim=10).posterior
    ),
    cuqi.sampler.MALA(
        target=cuqi.testproblem.Deconvolution1D(dim=10).posterior
    ),
    cuqi.sampler.NUTS(
        target=cuqi.testproblem.Deconvolution1D(dim=10).posterior
    ),
    cuqi.sampler.LinearRTO(
        target=cuqi.testproblem.Deconvolution1D(dim=10).posterior
    ),
    cuqi.sampler.RegularizedLinearRTO(
        target=create_regularized_target(dim=16)
    ),
    cuqi.sampler.UGLA(target=create_lmrf_prior_target(dim=16)),
    cuqi.sampler.Direct(target=cuqi.distribution.Gaussian(np.zeros(10), 1)),
    cuqi.sampler.Conjugate(target=create_conjugate_target("Gaussian-Gamma")),
    cuqi.sampler.ConjugateApprox(
        target=create_conjugate_target("LMRF-Gamma")
    ),
    cuqi.sampler.MYULA(target=create_myula_target(dim=16)),
    cuqi.sampler.PnPULA(target=create_myula_target(dim=16)),
    cuqi.sampler.HybridGibbs(
        target=HybridGibbs_target_1(),
        sampling_strategy={
            "x" : cuqi.sampler.NUTS(max_depth=7),
            "s" : cuqi.sampler.Conjugate()
        },
    ),
]

@pytest.mark.parametrize("sampler", callback_testing_sampler_instances)
def test_passing_callback_to_sampler(sampler):
    """ Test that the callback function is called by the sampler and
    the sampler passes the correct arguments to the callback function. """

    Ns = 10
    indices = [] # list to store the indices of the samples

    # Define the callback function
    def callback_function(callback_sampler, sample_index, num_samples):
        assert isinstance(callback_sampler, sampler.__class__)
        assert num_samples == Ns
        indices.append(sample_index)

    # Create a sampler with callback function
    if not isinstance(sampler, cuqi.sampler.HybridGibbs):
        my_sampler = sampler.__class__(sampler.target, callback=callback_function)
    else:
        my_sampler = cuqi.sampler.HybridGibbs(
            sampler.target,
            sampling_strategy={
                "x": cuqi.sampler.NUTS(max_depth=7),
                "s": cuqi.sampler.Conjugate(),
            },
            callback=callback_function,
        )

    # Sample Ns warmup samples
    my_sampler.warmup(Ns)
    assert np.allclose(indices, np.arange(Ns))

    # reset the indices
    indices = []

    # Sample Ns samples
    my_sampler.sample(Ns)
    assert np.allclose(indices, np.arange(Ns))

def test_all_samplers_that_should_be_tested_for_callback_are_in_the_tested_list():
    """ Test that all samplers that should be tested for callback are in the callback_testing_sampler_instances. """
    # The classes of the tested samplers:
    tested_classes = [sampler.__class__ for sampler in callback_testing_sampler_instances]
    for cls in callback_testing_sampler_classes:
        assert cls in tested_classes, f"Sampler {cls} is not tested for callback."

# ============= End testing sampler callback =============
def test_gibbs_random_scan_order():
    target = HybridGibbs_target_1()
    sampling_strategy={
                "x": cuqi.sampler.LinearRTO(),
                "s": cuqi.sampler.Conjugate(),
            }
    
    sampler = cuqi.sampler.HybridGibbs(target, sampling_strategy, scan_order='random')
    np.random.seed(0)
    scan_order1 = sampler.scan_order
    scan_order2 = sampler.scan_order
    assert scan_order1 != scan_order2

def test_gibbs_scan_order():
    target = HybridGibbs_target_1()
    sampling_strategy={
                "x": cuqi.sampler.LinearRTO(),
                "s": cuqi.sampler.Conjugate(),
            }
    
    sampler = cuqi.sampler.HybridGibbs(target, sampling_strategy, scan_order=['x', 's'])
    assert sampler.scan_order == ['x', 's']

def test_online_thinning_with_mala_and_rto():

    # Define LinearModel and data
    A, y_obs, _ = cuqi.testproblem.Deconvolution1D().get_components()

    # Define Bayesian Problem
    x = cuqi.distribution.GMRF(np.zeros(A.domain_dim), 100)
    y = cuqi.distribution.Gaussian(A@x, 0.01**2)
    posterior = cuqi.distribution.JointDistribution(x, y)(y=y_obs)

    # Set up MALA and RTO samplers
    sampler_mala_1 = cuqi.sampler.MALA(posterior, scale=0.0001)
    sampler_mala_2 = cuqi.sampler.MALA(posterior, scale=0.0001)
    sampler_rto_1 = cuqi.sampler.LinearRTO(posterior, maxit=1000, tol=1e-8)
    sampler_rto_2 = cuqi.sampler.LinearRTO(posterior, maxit=1000, tol=1e-8)

    # Sample MALA and RTO with fixed seed, but different online thinning Nt
    np.random.seed(0)
    samples_mala_1 = sampler_mala_1.sample(100,Nt=5).get_samples()
    np.random.seed(0)
    samples_mala_2 = sampler_mala_2.sample(100,Nt=1).get_samples()
    np.random.seed(0)
    samples_rto_1 = sampler_rto_1.sample(100,Nt=5).get_samples()
    np.random.seed(0)
    samples_rto_2 = sampler_rto_2.sample(100,Nt=1).get_samples()

    # Check that the samples are the same for MALA
    assert np.allclose(samples_mala_1.samples[:,0], samples_mala_2.samples[:,0], rtol=1e-8)
    assert np.allclose(samples_mala_1.samples[:,1], samples_mala_2.samples[:,5], rtol=1e-8)
    assert np.allclose(samples_mala_1.samples[:,2], samples_mala_2.samples[:,10], rtol=1e-8)
    # Check that the samples are the same for RTO
    assert np.allclose(samples_rto_1.samples[:,0], samples_rto_2.samples[:,0], rtol=1e-8)
    assert np.allclose(samples_rto_1.samples[:,1], samples_rto_2.samples[:,5], rtol=1e-8)
    assert np.allclose(samples_rto_1.samples[:,2], samples_rto_2.samples[:,10], rtol=1e-8)

def test_online_thinning_with_hybrid_gibbs():

    # example adapted from https://cuqi-dtu.github.io/CUQI-Book/chapter04/gibbs.html

    # Model and data
    A, y_data, _ = cuqi.testproblem.Deconvolution1D(phantom='sinc', noise_std=0.005, PSF_param=6).get_components()

    # Get dimension of signal
    n = A.domain_dim

    d = cuqi.distribution.Gamma(1, 1e-4)
    s = cuqi.distribution.Gamma(1, 1e-4)
    x = cuqi.distribution.GMRF(np.zeros(n), lambda d: d)
    y = cuqi.distribution.Gaussian(A@x, cov=lambda s: 1/s)

    # Create joint distribution
    joint = cuqi.distribution.JointDistribution(y, x, d, s)

    # Define posterior by conditioning on the data
    posterior = joint(y=y_data)

    # Define sampling strategies
    sampling_strategy_1 = {
        'x': cuqi.sampler.LinearRTO(),
        'd': cuqi.sampler.Conjugate(),
        's': cuqi.sampler.Conjugate()
    }
    sampling_strategy_2 = {
        'x': cuqi.sampler.LinearRTO(),
        'd': cuqi.sampler.Conjugate(),
        's': cuqi.sampler.Conjugate()
    }

    # Define Gibbs samplers
    sampler_1 = cuqi.sampler.HybridGibbs(posterior, sampling_strategy_1)
    sampler_2 = cuqi.sampler.HybridGibbs(posterior, sampling_strategy_2)

    # Run sampler with different online thinnning Nt
    np.random.seed(0)
    samples_1 = sampler_1.sample(20, Nt=5).get_samples()
    np.random.seed(0)
    samples_2 = sampler_2.sample(20).get_samples() # by default Nt=1

    # Compare samples
    assert np.allclose(samples_1['d'].samples[:, 0], samples_2['d'].samples[:, 0], rtol=1e-5)
    assert np.allclose(samples_1['d'].samples[:, 1], samples_2['d'].samples[:, 5], rtol=1e-5)
    assert np.allclose(samples_1['d'].samples[:, 2], samples_2['d'].samples[:, 10], rtol=1e-5)
    assert np.allclose(samples_1['s'].samples[:, 0], samples_2['s'].samples[:, 0], rtol=1e-5)
    assert np.allclose(samples_1['s'].samples[:, 1], samples_2['s'].samples[:, 5], rtol=1e-5)
    assert np.allclose(samples_1['s'].samples[:, 2], samples_2['s'].samples[:, 10], rtol=1e-5)
    assert np.allclose(samples_1['x'].samples[:, 0], samples_2['x'].samples[:, 0], rtol=1e-5)
    assert np.allclose(samples_1['x'].samples[:, 1], samples_2['x'].samples[:, 5], rtol=1e-5)
    assert np.allclose(samples_1['x'].samples[:, 2], samples_2['x'].samples[:, 10], rtol=1e-5)


@pytest.mark.parametrize("step_size", [None, 0.1])
@pytest.mark.parametrize("num_sampling_steps_x", [1, 5])
@pytest.mark.parametrize("nb", [5, 20])
def test_NUTS_within_Gibbs_consistant_with_NUTS(step_size, num_sampling_steps_x, nb):
    """ Test that using NUTS sampler within HybridGibbs sampler is consistant
    with using NUTS sampler alone for sampling and tuning. This test ensures 
    NUTS within HybridGibbs statefulness.
    """

    ns = 15 # number of sampling steps
    tune_freq = 0.1

    np.random.seed(0)
    # Forward problem
    A, y_data, info = cuqi.testproblem.Deconvolution1D(
        dim=5, phantom='sinc', noise_std=0.001).get_components()

    # Bayesian Inverse Problem
    x = cuqi.distribution.GMRF(np.zeros(A.domain_dim), 50)
    y = cuqi.distribution.Gaussian(A@x, 0.001**2)

    # Posterior
    target = cuqi.distribution.JointDistribution(y, x)(y=y_data)

    # Sample with NUTS within HybridGibbs
    np.random.seed(0)
    sampling_strategy = {
        "x" : cuqi.sampler.NUTS(max_depth=4, step_size=step_size)
    }

    num_sampling_steps = {
    "x" : num_sampling_steps_x
    }

    sampler_gibbs = cuqi.sampler.HybridGibbs(target,
                                                       sampling_strategy,
                                                       num_sampling_steps)
    sampler_gibbs.warmup(nb, tune_freq=tune_freq)
    sampler_gibbs.sample(ns)
    samples_gibbs = sampler_gibbs.get_samples()["x"].samples

    # Sample with NUTS alone
    np.random.seed(0)
    sampler_nuts = cuqi.sampler.NUTS(target,
                                               max_depth=4,
                                               step_size=step_size)
    # Warm up (when num_sampling_steps_x>0, we do not using built-in warmup
    #          in order to control number of steps between tuning steps to
    #          match Gibbs sampling behavior)
    if num_sampling_steps_x == 1:
        sampler_nuts.warmup(nb, tune_freq=tune_freq)
    else:
        tune_interval = max(int(tune_freq * nb), 1)
        for count in range(nb):
            for _ in range(num_sampling_steps_x):
                sampler_nuts.sample(1)
            if (count+1) % tune_interval == 0:
                sampler_nuts.tune(None, count//tune_interval)
    # Sample
    sampler_nuts.sample(ns * num_sampling_steps_x)
    samples_nuts = sampler_nuts.get_samples().samples
    # skip every num_sampling_steps_x samples to match Gibbs samples
    samples_nuts_skip = samples_nuts[:, num_sampling_steps_x - 1::num_sampling_steps_x]

    # assert warmup samples are correct:
    assert np.allclose(
        samples_gibbs[:, :nb],
        samples_nuts_skip[:, :nb],
        rtol=1e-5,
    )

    # assert samples are correct:
    assert np.allclose(
        samples_gibbs[:, nb:],
        samples_nuts_skip[:, nb:],
        rtol=1e-5,
    )

def test_enabling_FD_gradient_in_HybridGibbs_target():
    """Test enabling FD gradient in HybridGibbs target."""
    # Fix seed
    np.random.seed(0)
    # Multiple input model
    model = cuqi.model.Model(
        lambda a, b: a * b,
        domain_geometry=cuqi.geometry._ProductGeometry(
            cuqi.geometry.Discrete(["a"]), cuqi.geometry.Discrete(["b"])
        ),
        range_geometry=cuqi.geometry.Discrete(["c"]),
    )
    
    # Define Bayesian model and posterior distribution
    a = cuqi.distribution.Gaussian(mean=0, cov=1)
    b = cuqi.distribution.Gaussian(mean=0, cov=1)
    c = cuqi.distribution.Gaussian(mean=model(a, b), cov=0.1)
    J = cuqi.distribution.JointDistribution(a, b, c)

    posterior = J(c=4)
    
    # Gibbs sampling strategy with NUTS for both a and b
    sampling_strategy = {
        "a": cuqi.sampler.NUTS(),
        "b": cuqi.sampler.NUTS(),
    }

    # Attempt creating HybridGibbs sampler to warmup without gradient
    # implemented for the posterior should raise an error
    with pytest.raises(ValueError, match="Target must have logd and gradient methods"):
        sampler = cuqi.sampler.HybridGibbs(posterior, sampling_strategy)

    # Now enable FD gradient for both a and b in the posterior and create the
    # HybridGibbs sampler and run sampling 
    epsilon = {"a": 1e-8, "b": 1e-8}
    posterior.enable_FD(epsilon=epsilon)
    sampler = cuqi.sampler.HybridGibbs(posterior, sampling_strategy)
    sampler.warmup(20)
    sampler.sample(40)

    # Check that the samples gives the expected mean values for the given
    # seed
    assert sampler.get_samples()["a"].mean() == pytest.approx(2.099, rel=1e-2)
    assert sampler.get_samples()["b"].mean() == pytest.approx(1.859, rel=1e-2)