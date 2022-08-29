import numpy as np
import cuqi
import pytest

from cuqi import geometry


@pytest.mark.parametrize("is_par,plot_par",	[(True,False),(True,True),(False,False)])
@pytest.mark.parametrize("geom",[
						(cuqi.geometry.Discrete(1)),
						(cuqi.geometry.Discrete(1)),
						(cuqi.geometry.Discrete(3)),
						(cuqi.geometry.Continuous1D(1)),
						(cuqi.geometry.Continuous1D(1)),
						(cuqi.geometry.Continuous1D(3)),
						])
def test_samples_plot(geom,is_par,plot_par):
    dim = geom.par_dim
    x = cuqi.distribution.Normal(np.zeros(dim),np.ones(dim),geometry=geom)
    s = x.sample(10)

    #Verify plotting of single and multiple samples
    s.plot(is_par=is_par,plot_par=plot_par)
    s.plot(0,is_par=is_par,plot_par=plot_par)
    #s.plot_chain() #No default selection of chain
    s.plot_chain(0)
    if dim > 2:
        s.plot([0,2],is_par=is_par,plot_par=plot_par)
        s.plot_chain([0,2])

@pytest.mark.parametrize("kwargs",[
                        ({}),
                        ({"max_lag":10,"textsize":25}),
                        ])
def test_samples_plot_autocorrelation(kwargs):
    # Make basic distribution and sample
    dist = cuqi.distribution.DistributionGallery("CalSom91")
    sampler = cuqi.sampler.MetropolisHastings(dist)
    samples = sampler.sample_adapt(1000)

    # Switch to discrete geometry (easiest for "variable" names)
    samples.geometry = cuqi.geometry.Discrete(["alpha","beta"])

    # Plot with defaults
    samples.plot_autocorrelation()

    # Plot with defaults arguments
    samples.plot_autocorrelation(**kwargs)

    # Plot for single parameter + arguments
    samples.plot_autocorrelation([0],**kwargs)


@pytest.mark.parametrize("kwargs",[
                        ({}),
                        ({"combined":False,"tight_layout":False}),
                        ])
def test_samples_plot_trace(kwargs):
    # Make basic distribution and sample
    dist = cuqi.distribution.DistributionGallery("CalSom91")
    sampler = cuqi.sampler.MetropolisHastings(dist)
    samples = sampler.sample_adapt(1000)

    # Switch to discrete geometry (easiest for "variable" names)
    samples.geometry = cuqi.geometry.Discrete(["alpha","beta"])

    # Plot with defaults
    samples.plot_trace()

    # Plot with defaults arguments
    samples.plot_trace(**kwargs)

    # Plot for single parameter + arguments
    samples.plot_trace([0],**kwargs)

@pytest.mark.parametrize("kwargs",[
                        ({"marginals": False}),
                        ({"marginals": True}),
                        ({"kind":"kde","marginals":False}),
                        ({"kind":"kde","marginals":True}),
                        ({"kind":"hexbin","marginals":False}),
                        ({"kind":"hexbin","marginals":True}),
                        ])
def test_samples_plot_pair(kwargs):
    # Make basic distribution and sample
    dist = cuqi.distribution.Gaussian(np.array([1,2,3,4]),1)
    sampler = cuqi.sampler.MetropolisHastings(dist)
    samples = sampler.sample_adapt(1000)
    samples.geometry = cuqi.geometry.Discrete(["a","b","c","d"])

    # Plot with defaults
    samples.plot_pair()

    # Plot with defaults arguments
    samples.plot_pair(**kwargs)

    # Plot for single parameter + arguments
    ax = samples.plot_pair([1,2],**kwargs)

    # Check if the correct labels are passed in specific case
    if kwargs.get("marginals") == False:
        assert ax.get_xlabel() == "b"
        assert ax.get_ylabel() == "c"

def test_rhat_values():
    rng = np.random.RandomState(0)
    mean = 0; var  = 1
    samples1 = cuqi.distribution.Normal(mean,var).sample(20000,rng=rng)
    samples2 = cuqi.distribution.Normal(mean,var).sample(20000,rng=rng) 
    rhat_results = samples1.compute_rhat(samples2)
    assert np.allclose(rhat_results[0], 1, rtol=1e-3)

def test_rhat_geometry():
    mean = 0; var  = 1
    samples1 = cuqi.distribution.Normal(mean,var).sample(200)
    samples2 = cuqi.distribution.Normal(mean,var).sample(200) 
    samples1.geometry = cuqi.geometry.Discrete(["alpha","beta1"])
    samples2.geometry = cuqi.geometry.Discrete(["alpha","beta2"])
    with pytest.raises(TypeError): #Type error since geometry does not match.
        samples1.compute_rhat(samples2)

def test_ess():
    dist = cuqi.distribution.DistributionGallery("CalSom91")
    sampler = cuqi.sampler.MetropolisHastings(dist)
    samples = sampler.sample_adapt(500)
    assert samples.compute_ess().shape == samples.geometry.par_shape

@pytest.mark.parametrize("geometry", [cuqi.geometry.Discrete(2),
                                      cuqi.geometry.MappedGeometry(
                                          cuqi.geometry.Continuous1D(2), map=lambda x: x**2),
                                      cuqi.geometry.KLExpansion(np.arange(0, 1, .1))])
def test_samples_funvals(geometry):
    """Test that the function values are computed correctly."""
    Ns = 10
    samples = cuqi.samples.Samples(
        np.random.randn(geometry.par_dim, Ns), geometry=geometry)

    funvals = np.empty((geometry.par_dim, Ns))
    for i, s in enumerate(samples):
        funvals[:, i] = geometry.par2fun(s)

    assert np.allclose(samples.funvals.samples, funvals)


@pytest.mark.parametrize("percent", [10, 50, 90, 95, 99])
@pytest.mark.parametrize("compute_on_par", [False, True])
@pytest.mark.parametrize("geometry", [cuqi.geometry.Discrete(2),
                                      cuqi.geometry.MappedGeometry(
                                        cuqi.geometry.Continuous1D(2), map=lambda x: x**2)])
def test_compute_ci(percent, compute_on_par, geometry):
    dist = cuqi.distribution.DistributionGallery("CalSom91")
    sampler = cuqi.sampler.MetropolisHastings(dist)
    par_samples = sampler.sample_adapt(500)
    par_samples.geometry = geometry
    
    samples = par_samples if compute_on_par else par_samples.funvals
    ci = samples.compute_ci(percent)

    # manually compute ci
    lb = (100-percent)/2
    up = 100-lb
    lo_conf, up_conf = np.percentile(samples.samples, [lb, up], axis=-1)

    assert np.allclose(ci[0], lo_conf)
    assert np.allclose(ci[1], up_conf)

@pytest.mark.parametrize("is_par", [False, True, None]) #passing is_par will raise an error.
@pytest.mark.parametrize("plot_par, compute_on_par",
                         [(True, True),
                          (True, False), # This case will raise an error.
                          (False, True),
                          (False, False)])
@pytest.mark.parametrize("geometry", [cuqi.geometry.Discrete(2),
                                      cuqi.geometry.KLExpansion(np.arange(0, 1, .1))])
def test_plot_ci_par_func(is_par, plot_par, compute_on_par, geometry):
    """Test passing flags to plot_ci."""
    np.random.seed(0)
    par_samples = cuqi.samples.Samples(np.random.randn(geometry.par_dim, 10), geometry=geometry)
    samples = par_samples if compute_on_par else par_samples.funvals

    if is_par is not None:
        # User should not be able to pass is_par for plotting ci because it is
        #  determined automatically depending on self.is_par and plot_par
        with pytest.raises(ValueError):
            samples.plot_ci(is_par=is_par, plot_par=plot_par)

    elif plot_par and not compute_on_par:
        # User cannot ask for computing statistics on function values then plotting on parameter space
        # plot_ci will raise an error in this case
        with pytest.raises(ValueError):
            samples.plot_ci(plot_par=plot_par)
    else:
        #The remaining cases should not raise an error.
        import matplotlib.pyplot as plt
        plt.figure()
        samples.plot_ci(plot_par=plot_par)
