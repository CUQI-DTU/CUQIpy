import cuqi
import numpy as np
import pytest

def test_RegularizedGaussian_default_init():
    """ Test that the implicit regularized Gaussian requires at least 1 regularization argument """

    with pytest.raises(ValueError, match="At least some "):
        x = cuqi.implicitprior.RegularizedGaussian(np.zeros(5), 1)

def test_RegularizedGaussian_guarding_statements():
    """ Test that we catch incorrect initialization of RegularizedGaussian """

    # More than 1 argument
    with pytest.raises(ValueError, match="User-defined proximals and "):
        cuqi.implicitprior.RegularizedGaussian(np.zeros(5), 1, proximal=lambda s,z: s, constraint="nonnegativity")

    # Proximal
    with pytest.raises(ValueError, match="Proximal needs to be callable or a list. See documentation."):
        cuqi.implicitprior.RegularizedGaussian(np.zeros(5), 1, proximal=1)

    with pytest.raises(ValueError, match="Proximal should take 2 arguments"):
        cuqi.implicitprior.RegularizedGaussian(np.zeros(5), 1, proximal=lambda s: s)

    # Projector
    with pytest.raises(ValueError, match="Projector needs to be callable"):
        cuqi.implicitprior.RegularizedGaussian(np.zeros(5), 1, projector=1)

    with pytest.raises(ValueError, match="Projector should take 1 argument"):
        cuqi.implicitprior.RegularizedGaussian(np.zeros(5), 1, projector=lambda s,z: s)

def test_creating_restorator():
    """ Test creating the object from restorator class."""

    def func(x, restoration_strength=0.1):
        return x, True
    restorator = cuqi.implicitprior.RestorationPrior(func)
    assert np.allclose(restorator.restore(np.ones(4), 0.1), np.ones(4))
    assert restorator.info == True

def test_handling_invalid_restorator():
    """ Test handling invalid restorator."""
    # Invalid return type 1: None
    def func_1(x, restoration_strength=0.1):
        return
    restore_prior_1 = cuqi.implicitprior.RestorationPrior(func_1)
    with pytest.raises(ValueError, match=r"Unsupported return type .*"):
        restore_prior_1.restore(np.ones(4), 0.1)
    # Invalid return type 2: one parameter
    def func_2(x, restoration_strength=0.1):
        return x
    restore_prior_2 = cuqi.implicitprior.RestorationPrior(func_2)
    with pytest.raises(ValueError, match=r"Unsupported return type .*"):
        restore_prior_2.restore(np.ones(4), 0.1)
    # Invalid return type 3: tuple with 3 elements
    def func_3(x, restoration_strength=0.1):
        return x, None, False
    restore_prior_3 = cuqi.implicitprior.RestorationPrior(func_3)
    with pytest.raises(ValueError, match=r"Unsupported return type .*"):
        restore_prior_3.restore(np.ones(4), 0.1)
    # Invalid return type 4: list with 2 elements
    def func_4(x, restoration_strength=0.1):
        return [x, None]
    restore_prior_4 = cuqi.implicitprior.RestorationPrior(func_4)
    with pytest.raises(ValueError, match=r"Unsupported return type .*"):
        restore_prior_4.restore(np.ones(4), 0.1)

def test_creating_restorator_with_potential():
    """ Test creating the object from restorator class with a potential."""

    def func(x, restoration_strength=1):
        return x/(1+restoration_strength), True
    def potential(x):
        return (x**2).sum()/2
    restorator = cuqi.implicitprior.RestorationPrior(restorator=func, potential=potential)
    assert np.allclose(restorator.restore(np.ones(1), restoration_strength=1), np.ones(1)/(1+1))
    assert restorator.info == True
    assert restorator.logpdf(np.ones(4)) == -2
    

def test_creating_moreau_yoshida_prior_gradient():
    """ Test creating MoreauYoshidaPrior."""

    def func(x, restoration_strength=1):
        return x/(1+restoration_strength), True
    def potential(x):
        return (x**2).sum()/2
    restorator = cuqi.implicitprior.RestorationPrior(func,
                                                    potential=potential)
    myprior = cuqi.implicitprior.MoreauYoshidaPrior(restorator, smoothing_strength=0.1)
    assert np.allclose(myprior.gradient(np.ones(1)), -np.ones(1)/(1+myprior.smoothing_strength))
    assert myprior.logpdf(np.ones(1)) == -0.5*myprior.smoothing_strength/(1+myprior.smoothing_strength)
            
def test_ConstrainedGaussian_alias():
    """ Test that the implicit constrained Gaussian is a correct allias for an implicit regularized Gaussian """

    x = cuqi.implicitprior.ConstrainedGaussian(np.zeros(5), 1, constraint="nonnegativity")

    assert isinstance(x, cuqi.implicitprior.RegularizedGaussian)
    assert x.preset["constraint"] == "nonnegativity"
    assert x.preset["regularization"] is None

def test_NonnegativeGaussian_alias():
    """ Test that the implicit nonnegative Gaussian is a correct allias for an implicit regularized Gaussian """

    x = cuqi.implicitprior.NonnegativeGaussian(np.zeros(5), 1)

    assert isinstance(x, cuqi.implicitprior.RegularizedGaussian)
    assert x.preset["constraint"] == "nonnegativity"
    assert x.preset["regularization"] is None

def test_ConstrainedGMRF_alias():
    """ Test that the implicit constrained GMRF is a correct allias for an implicit regularized GMRF """

    x = cuqi.implicitprior.ConstrainedGMRF(np.zeros(5), 1, constraint="nonnegativity")

    assert isinstance(x, cuqi.implicitprior.RegularizedGMRF)
    assert x.preset["constraint"] == "nonnegativity"
    assert x.preset["regularization"] is None

def test_NonnegativeGMRF_alias():
    """ Test that the implicit nonnegative GMRF is a correct allias for an implicit regularized GMRF """

    x = cuqi.implicitprior.NonnegativeGMRF(np.zeros(5), 1)

    assert isinstance(x, cuqi.implicitprior.RegularizedGMRF)
    assert x.preset["constraint"] == "nonnegativity"
    assert x.preset["regularization"] is None

def test_RegularizedUnboundedUniform_is_RegularizedGaussian():
    """ Test that the implicit regularized unbounded uniform create a Regularized Gaussian with zero sqrtprec """
    # NOTE: Test is based on the current assumption that the regularized uniform is modeled as a Gaussian with zero precision. This might change in the future.

    x = cuqi.implicitprior.RegularizedUnboundedUniform(cuqi.geometry.Continuous1D(5), regularization="l1", strength = 5.0)
    
    assert np.allclose(x.gaussian.sqrtprec, 0.0)

def test_RegularizedGaussian_conditioning_constrained():
    """ Test that conditioning the implicit regularized Gaussian works as expected """
    
    x  = cuqi.implicitprior.RegularizedGMRF(lambda a:a*np.ones(2**2),
                     prec = lambda b:5*b,
                     constraint = "nonnegativity",
                     geometry = cuqi.geometry.Image2D((2,2)))
    
    assert x.get_mutable_variables() == ['mean', 'prec']
    assert x.get_conditioning_variables() == ['a', 'b']

    x = x(a=1, b=2)

    assert np.allclose(x.mean, [1, 1, 1, 1])
    assert np.allclose(x.prec, 10)

def test_RegularizedGaussian_conditioning_strength():
    """ Test that conditioning the implicit regularized Gaussian works as expected """
    
    x  = cuqi.implicitprior.RegularizedGMRF(lambda a:a*np.ones(2**2),
                     prec = lambda b:5*b,
                     regularization = "tv",
                     strength = lambda c:c*2,
                     geometry = cuqi.geometry.Image2D((2,2)))
    
    assert x.get_mutable_variables() == ['mean', 'prec', 'strength']
    assert x.get_conditioning_variables() == ['a', 'b', 'c']

    x = x(a=1, b=2, c=3)

    assert np.allclose(x.mean, [1, 1, 1, 1])
    assert np.allclose(x.prec, 10)
    assert np.allclose(x.strength, 6)

def test_RegularizedGaussian_double_preset():
    """ Test that the implicit RegularizedGaussian can handle combined regularization and constraint presets """

    constraint = "nonnegativity"
    regularization = "tv"
    x = cuqi.implicitprior.RegularizedGaussian(np.zeros(5), 1,
                                                regularization = regularization, strength = 5,
                                                constraint = constraint)

    # Check that the correct presets are set
    assert x.preset["constraint"] == constraint
    assert x.preset["regularization"] == regularization
    # Check whether the constructed proximal list is of correct size
    assert len(x.proximal) == 2
    assert len(x.proximal[0]) == 2
    assert len(x.proximal[1]) == 2