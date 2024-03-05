import cuqi
import pytest
import numpy as np

def assert_true_if_sampling_is_equivalent(sampler_old, sampler_new, Ns=100, atol=1e-1):
    """ Assert that the samples from the old and new sampler are equivalent. """

    np.random.seed(0)
    samples_old = sampler_old.sample(Ns).samples

    np.random.seed(0)
    samples_new = sampler_new.sample(Ns).get_samples().samples[:-1].T

    # ToDo: Add regression test for warmup/burnin
    #samples_old = cuqi.sampler.MH(target).sample_adapt(N=Ns, Nb=Nb).samples
    #samples_new = cuqi.mcmc.MH_new(target).warmup(Nb).sample(Ns).get_samples().samples[Nb:-1].T

    assert np.allclose(samples_old, samples_new, atol=atol), f"Old: {samples_old}\nNew: {samples_new}"


@pytest.mark.parametrize("target", 
    [
        cuqi.testproblem.Deconvolution1D().posterior
    ])
def test_MH_regression_sample(target: cuqi.density.Density):
    """Test the MH sampler regression."""
    sampler_old = cuqi.sampler.MH(target, scale=1)
    sampler_new = cuqi.mcmc.MH_new(target, scale=1)
    assert_true_if_sampling_is_equivalent(sampler_old, sampler_new)


@pytest.mark.parametrize("target", 
    [
        cuqi.testproblem.Deconvolution1D().posterior
    ])
def test_pCN_regression_sample(target: cuqi.density.Density):
    """Test the pCN sampler regression."""
    sampler_old = cuqi.sampler.pCN(target, scale=1)
    sampler_new = cuqi.mcmc.PCN_new(target, scale=1)
    assert_true_if_sampling_is_equivalent(sampler_old, sampler_new)


@pytest.mark.parametrize("target", 
    [
        cuqi.testproblem.Deconvolution1D().posterior
    ])
def test_MALA_regression_sample(target: cuqi.density.Density):
    """Test the pCN sampler regression."""
    sampler_old = cuqi.sampler.MALA(target, scale=1)
    sampler_new = cuqi.mcmc.MALA_new(target, scale=1)
    assert_true_if_sampling_is_equivalent(sampler_old, sampler_new)


