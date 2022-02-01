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
    dim = geom.dim
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