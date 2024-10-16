import numpy as np
import cuqi
import pytest
import matplotlib
import matplotlib.pyplot as plt
from cuqi import geometry
from cuqi.samples import Samples

@pytest.mark.parametrize("to_funvals, plot_par",
    [(True,False),(True,True),(False,False),(False,True)])
@pytest.mark.parametrize("geom",[
						(cuqi.geometry.Discrete(1)),
						(cuqi.geometry.Discrete(3)),
						(cuqi.geometry.Continuous1D(1)),
						(cuqi.geometry.Continuous1D(3)),
                        (cuqi.geometry.Image2D((3,3))),
						])
def test_samples_plot(geom, to_funvals, plot_par):
    # Make basic distribution and sample
    dim = geom.par_dim
    x = cuqi.distribution.Normal(np.zeros(dim),np.ones(dim),geometry=geom)
    s = x.sample(10)

    # Convert to funvals if requested
    if to_funvals:
        s = s.funvals

    # Verify that asking for plotting in the parameter space when samples
    # are funvals raises error
    if to_funvals and plot_par:
        with pytest.raises(Exception, match=r"Plot par is true, but is_par is false"):
            s.plot(plot_par=plot_par)
        return
    
    #Verify plotting of single and multiple samples
    s.plot(plot_par=plot_par)
    s.plot(0,plot_par=plot_par)

    # Raise error when calling plot_chain if the samples are not in vector form
    if not s.is_vec:
        with pytest.raises(Exception, match=r"Cannot perform plot_chain"):
            s.plot_chain()
        # Convert the samples to vector form
        s = s.vector
    
    #s.plot_chain() #No default selection of chain
    s.plot_chain(0)
    if dim > 2:
        s.plot([0,2],plot_par=plot_par)
        s.plot_chain([0,2])
    
    # Verify passing is_par raises error (because is_par is determined by
    # the type of the samples)
    with pytest.raises(ValueError,match=r"Cannot pass is_par as a plotting argument"):
        s.plot(is_par=not to_funvals,plot_par=plot_par)

@pytest.mark.parametrize("geom",[
						(cuqi.geometry.Discrete(1)),
						(cuqi.geometry.Continuous1D(1))
						])
@pytest.mark.parametrize("kwargs",[
                        ({}),
                        ({"marker":"*"}),
                        ])
def test_1D_samples_plot(geom, kwargs):
    # Make basic distribution and sample
    my_gaussian = cuqi.distribution.Gaussian(mean=1, cov=1)
    samples = my_gaussian.sample(10)
    samples.geometry = geom
    fig1 = samples.plot(**kwargs)
    fig2 = samples.plot(0, **kwargs)

@pytest.mark.parametrize("to_funvals, plot_par", [(True,False),(True,True),(False,False),(False,True)])
@pytest.mark.parametrize("plot_method, plot_par_supported, arviz",
                         [(Samples.plot_autocorrelation, False, True),
                          (Samples.plot_trace, False, True),
                          (Samples.plot_violin, False, True),
                          (Samples.plot_pair, False, True),
                          (Samples.plot_std, True, False),
                          (Samples.plot_mean, True, False),
                          (Samples.plot_median, True, False),
                          (Samples.plot_variance, True, False),
                          (Samples.plot_ci_width, True, False)])
@pytest.mark.parametrize("geom",[
						(cuqi.geometry.Discrete(3)),
						(cuqi.geometry.Continuous1D(4)),
                        (cuqi.geometry.Image2D((4,4))),
                        (cuqi.geometry.MappedGeometry(
                             cuqi.geometry.Image2D((3,3)),
                             map=lambda x: x**2+2))
						])
def test_samples_plot_methods(geom, to_funvals, plot_par,
                              plot_method, plot_par_supported,
                              arviz):
    """Test Samples class plotting methods"""

    # Make basic distribution and sample
    np.random.seed(0)
    dim = geom.par_dim
    x = cuqi.distribution.Normal(np.zeros(dim),np.ones(dim),geometry=geom)
    s = x.sample(10)

    # Convert to funvals if requested
    if to_funvals:
        s = s.funvals

    # Verify that using an arviz based method when the samples are not in
    # vector form raises error
    if arviz and not s.is_vec:
        with pytest.raises(Exception, match=r"Samples are not in a vector representation"):
            plot_method(s)
        # Convert the samples to vector form
        s = s.vector
  
    # Verify asking for plot_par when samples are funvals raises error
    if to_funvals and plot_par:
        if plot_par_supported:
            # The case of plotting methods that support plot_par
            with pytest.raises(Exception, match=r"Plot par is true, but is_par is false"):
                plot_method(s, plot_par=plot_par)
            return
        else:
            # The case of plotting methods that do not support plot_par
            with pytest.raises(TypeError, match=r"got an unexpected keyword argument"):
                plot_method(s, plot_par=plot_par)
            return
    
    # Verify passing is_par raises error
    if plot_par_supported:
        with pytest.raises(ValueError,match=r"Cannot pass is_par as a plotting argument"):
            plot_method(s, is_par=not to_funvals)
    else:
        with pytest.raises(TypeError,match=r"got an unexpected keyword argument"):
            plot_method(s, is_par=not to_funvals)
    
    #Verify plotting method works
    if not plot_par_supported:
        plot_method(s)
    else:
        plot_method(s, plot_par=plot_par)


@pytest.mark.parametrize("kwargs",[
                        ({}),
                        ({"max_lag":10,"textsize":25}),
                        ])
def test_samples_plot_autocorrelation(kwargs):
    # Make basic distribution and sample
    dist = cuqi.distribution.DistributionGallery("CalSom91")
    sampler = cuqi.sampler.MH(dist)
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
                        ({"exact":[1, 1]}) # Check exact can be compared with
                        ])
def test_samples_plot_trace(kwargs):
    # Make basic distribution and sample
    dist = cuqi.distribution.DistributionGallery("CalSom91")
    sampler = cuqi.sampler.MH(dist)
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
    sampler = cuqi.sampler.MH(dist)
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
    sampler = cuqi.sampler.MH(dist)
    samples = sampler.sample_adapt(500)
    assert samples.compute_ess().shape == samples.geometry.par_shape

@pytest.mark.parametrize("geometry", [cuqi.geometry.Discrete(2),
                                      cuqi.geometry.MappedGeometry(
                                          cuqi.geometry.Continuous1D(2), map=lambda x: x**2),
                                      cuqi.geometry.KLExpansion(np.arange(0, 1, .1), num_modes=4),
                                      cuqi.geometry.Image2D((3,3)),
                                      cuqi.geometry.MappedGeometry(
                                          cuqi.geometry.Image2D((4,3)), map=lambda x: x**2+1)])
def test_samples_funvals(geometry):
    """Test that the function values are computed correctly."""
    # Generate some random samples
    Ns = 10
    samples = cuqi.samples.Samples(
        np.random.randn(geometry.par_dim, Ns), geometry=geometry)

    # Compute the function values
    funvals_vec = np.empty((geometry.funvec_dim, Ns))
    for i, s in enumerate(samples):
        funvals_vec[:, i] = geometry.fun2vec(geometry.par2fun(s))

    # Check that the funvals property is implemented correctly
    assert np.allclose(samples.funvals.vector.samples, funvals_vec)


@pytest.mark.parametrize("percent", [10, 50, 90, 95, 99])
@pytest.mark.parametrize("compute_on_par", [False, True])
@pytest.mark.parametrize("geometry", [cuqi.geometry.Discrete(2),
                                      cuqi.geometry.MappedGeometry(
                                        cuqi.geometry.Continuous1D(2), map=lambda x: x**2),
                                      cuqi.geometry.Image2D((2,1)),
                                      cuqi.geometry.MappedGeometry(
                                          cuqi.geometry.Image2D((1,2)), map=lambda x: x**2+1)])
def test_compute_ci(percent, compute_on_par, geometry):
    dist = cuqi.distribution.DistributionGallery("CalSom91")
    sampler = cuqi.sampler.MH(dist)
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


@pytest.mark.parametrize("geometry",
                         [cuqi.geometry.Discrete(4),
                          cuqi.geometry.KLExpansion(np.linspace(0, 1, 4)),
                          cuqi.geometry.Continuous2D((2, 2)),
                          cuqi.geometry.Image2D((2, 2))])
@pytest.mark.parametrize("plot_par", [False, True])
def test_plot_ci_returned_values(geometry, plot_par):
    """Test that the correct matplotlib object types are returned by plot_ci."""
    # Create samples and plot ci
    np.random.seed(0)
    samples = cuqi.samples.Samples(np.random.randn(
        geometry.par_dim, 10), geometry=geometry)
    plot_objs = samples.plot_ci(
        100, plot_par=plot_par, exact=np.random.randn(geometry.par_dim))

    # Check that the correct matplotlib object types are returned:
    # Continuous 2D plots
    if isinstance(geometry, cuqi.geometry.Continuous2D) and not plot_par: 
        for obj in plot_objs:
            assert isinstance(obj, matplotlib.collections.PolyCollection)

    # Image 2D plots
    elif isinstance(geometry, cuqi.geometry.Image2D) and not plot_par: 
        for obj in plot_objs:
            assert isinstance(obj, matplotlib.image.AxesImage)

    # Discrete plots
    elif plot_par or isinstance(geometry, cuqi.geometry.Discrete):
        assert isinstance(plot_objs[0], matplotlib.lines.Line2D)
        assert isinstance(plot_objs[1], matplotlib.lines.Line2D)
        assert isinstance(plot_objs[2], matplotlib.container.ErrorbarContainer)

    # Continuous 1D plots
    elif plot_par or isinstance(geometry, cuqi.geometry.Discrete):  
        assert isinstance(plot_objs[0], matplotlib.lines.Line2D)
        assert isinstance(plot_objs[1], matplotlib.lines.Line2D)
        assert isinstance(plot_objs[2], matplotlib.collections.PolyCollection)


@pytest.mark.parametrize("geom, map, imap, supported",
                         [(cuqi.geometry.Discrete(4), lambda x:x**2, lambda x:np.sqrt(x), True),
                          (cuqi.geometry.Continuous1D(15),
                           lambda x:x+12, lambda x:x-12, True),
                             (cuqi.geometry.Image2D((4, 5)), lambda x:x **
                              2+1, lambda x:np.sqrt(x-1), True)
                          ])
def test_parameters_property(geom, map, imap, supported):
    """Test that the Samples parameters property is computed correctly. And that an error is generated
    when the Samples geometry type does not support computing funvals."""
    # Create random samples:
    np.random.seed(0)
    Ns = 10
    mapped_geom = cuqi.geometry.MappedGeometry(geom, map, imap)
    val = np.absolute(np.random.rand(mapped_geom.par_dim, Ns))

    # Create Samples object:
    samples = cuqi.samples.Samples(val, geometry=mapped_geom)

    if not supported:
        with pytest.raises(ValueError, match=r"Cannot convert the samples to function values"):
            funvals = samples.funvals
    else:
        # Compute function values and from the function values
        # compute the parameters:
        funvals_vec = samples.funvals.vector
        parameters = funvals_vec.funvals.parameters

        # Assert that the parameters and the function values (in vector 
        # representation) are different and that extracting the function values
        # and going back to the parameters is done correctly.
        assert not np.allclose(parameters.samples, funvals_vec.samples) and\
            np.allclose(parameters.samples, val)

def test_cuqiarray_default_geometry():
    """ Test that CUQIarray creates a default geometry when no geometry is passed"""
    v = cuqi.array.CUQIarray([0,1,2,3,4,5,6,7,8])
    assert type(v.geometry) is cuqi.geometry._geometry._DefaultGeometry1D

def test_cuqiarray_multidim():
    X = np.array([[1, 2], [3,4]])
    with pytest.raises(Exception) as e:
        C = cuqi.array.CUQIarray(X)
    assert "input_array cannot be multidimensional when initializing CUQIarray as parameter (with is_par True)." in str(e.value) # this message
    assert e.type == ValueError  

def test_cuqiarray_ispar_false_without_geometry():
    with pytest.raises(Exception) as e:
        X = cuqi.array.CUQIarray([1,2,3], is_par=False)
    assert "geometry cannot be none when initializing a CUQIarray as function values (with is_par False)." in str(e.value) # this message
    assert e.type == ValueError   

def test_violin_plot():
    """ Test that the violin plot is generated correctly. """
    dist = cuqi.distribution.DistributionGallery("CalSom91")
    sampler = cuqi.sampler.MH(dist)
    samples = sampler.sample_adapt(1000)
    samples.geometry = cuqi.geometry.Discrete(["alpha","beta"])

    ax = samples.plot_violin(shade=0.1)

    assert ax[0].get_title() == "alpha"
    assert ax[1].get_title() == "beta"

@pytest.mark.parametrize("method", [Samples.mean,
                                    Samples.variance,
                                    Samples.std,
                                    Samples.std,
                                    Samples.ci_width])
def test_computing_statistics_on_arbitrary_objects_raises_error(method):
    """ Test that computing statistics on arbitrary objects (np.matrix object for example) raises an error. """

    samples = cuqi.samples.Samples([object(), object()])
    with pytest.raises(TypeError, match=r"Cannot compute statistics"):
        method(samples)
