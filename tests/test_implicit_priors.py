import cuqi
import numpy as np
import pytest

def test_RegularizedGaussian_default_init():
    """ Test that the implicit regularized Gaussian requires at least 1 regularization argument """

    with pytest.raises(ValueError, match="Precisely one of "):
        x = cuqi.implicitprior.RegularizedGaussian(np.zeros(5), 1)

def test_RegularizedGaussian_guarding_statements():
    """ Test that we catch incorrect initialization of RegularizedGaussian """

    # More than 1 argument
    with pytest.raises(ValueError, match="Precisely one of "):
        cuqi.implicitprior.RegularizedGaussian(np.zeros(5), 1, proximal=lambda s,z: s, constraint="nonnegativity")

    # Proximal
    with pytest.raises(ValueError, match="Proximal needs to be callable"):
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
    assert np.allclose(restorator.restorate(np.ones(4)), np.ones(4))
    assert restorator.info == True


def test_creating_restorator_with_potential():
    """ Test creating the object from restorator class with a potential."""

    def func(x, restoration_strength=1):
        return x/(1+restoration_strength), True
    def potential(x):
        return (x**2).sum()/2
    restorator = cuqi.implicitprior.RestorationPrior(restorator=func, potential=potential, restoration_strength=1)
    assert np.allclose(restorator.restorate(np.ones(1)), np.ones(1)/(1+restorator.restoration_strength))
    assert restorator.info == True
    assert restorator.logpdf(np.ones(4)) == -2
    

def test_creating_moreau_yoshida_prior_gradient():
    """ Test creating MoreauYoshidaPrior."""

    def func(x, restoration_strength=1):
        return x/(1+restoration_strength), True
    def potential(x):
        return (x**2).sum()/2
    restorator = cuqi.implicitprior.RestorationPrior(func, restoration_strength=0.1,
                                                     potential=potential)
    myprior = cuqi.implicitprior.MoreauYoshidaPrior(restorator, smoothing_strength=0.1)
    assert np.allclose(myprior.smoothing_strength, restorator.restoration_strength)
    assert np.allclose(myprior.gradient(np.ones(1)), -np.ones(1)/(1+myprior.smoothing_strength))
    assert myprior.logpdf(np.ones(1)) == -0.5*myprior.smoothing_strength/(1+myprior.smoothing_strength)
    
def test_mismatch_smoothing_strength_restoration_strength_raises_error():
    """ Test that rises an error when smoothing_strength of the MoreauYoshidaPrior
    is not equal to restoration_strength in the restorator."""

    def func(x, restoration_strength=1):
        return x/(1+restoration_strength), True
    restorator = cuqi.implicitprior.RestorationPrior(func, restoration_strength=0.1)
    with pytest.raises(ValueError, 
                       match=r"must be equal to restoration_strength"):
        myprior = cuqi.implicitprior.MoreauYoshidaPrior(restorator, smoothing_strength=0.2)
        
@pytest.mark.parametrize("restoration_strength",[0.1, None, 0.09999999999999999])
def test_compatible_values_of_smoothing_strength_restoration_strength(restoration_strength):
    def func(x, restoration_strength=1):
        return x/(1+restoration_strength), True
    restorator = cuqi.implicitprior.RestorationPrior(func, restoration_strength=restoration_strength)
    myprior = cuqi.implicitprior.MoreauYoshidaPrior(restorator, smoothing_strength=0.1)

    
