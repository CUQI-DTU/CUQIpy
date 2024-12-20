# %%
import cuqi
import pytest
import numpy as np
from cuqi.geometry import Continuous1D, Discrete, MappedGeometry
from cuqi.experimental.geometry import _ProductGeometry

@pytest.fixture
def product_geometry():
    """fixture function to create a product geometry"""
    geometry1 = Continuous1D(np.linspace(0, 1, 100))
    geometry2 = MappedGeometry(
        Discrete(["sound_speed"]), map=lambda x: x**2, imap=lambda x: np.sqrt(x)
    )
    product_geometry = _ProductGeometry(geometry1, geometry2)
    return product_geometry

def test_par2fun_single_sample(product_geometry):
    """Test par2fun for single sample"""
    geometry1 = product_geometry.geometries[0]
    geometry2 = product_geometry.geometries[1]

    par1 = np.ones(geometry1.par_dim+geometry2.par_dim)
    par1[geometry1.par_dim:] = 2
    par2 = (np.ones(geometry1.par_dim), np.ones(geometry2.par_dim)*2)

    fun1 = product_geometry.par2fun(par1)
    fun2 = product_geometry.par2fun(par2[0], par2[1])
    fun2_a = geometry1.par2fun(par2[0])
    fun2_b = geometry2.par2fun(par2[1])

    assert np.allclose(len(fun1), 2)
    assert np.allclose(len(fun1), len(fun2))
    assert np.allclose(fun1[0], fun2[0])
    assert np.allclose(fun1[1], fun2[1])
    assert np.allclose(fun2[0], fun2_a)
    assert np.allclose(fun2[1], fun2_b)

def test_par2fun_multiple_samples(product_geometry):
    """Test par2fun for multiple samples"""
    geometry1 = product_geometry.geometries[0]
    geometry2 = product_geometry.geometries[1]
    
    # Number of samples is 5
    Ns = 5
    par3 = np.ones((geometry1.par_dim+geometry2.par_dim, Ns))
    par3[geometry1.par_dim:,:] = 2
    par4 = (np.ones((geometry1.par_dim, Ns)), 2*np.ones((geometry2.par_dim, Ns)))

    fun3 = product_geometry.par2fun(par3)
    fun4 = product_geometry.par2fun(par4[0], par4[1])
    fun4_a = geometry1.par2fun(par4[0])
    fun4_b = geometry2.par2fun(par4[1])

    assert(np.allclose(len(fun3), len(fun4)))
    assert(np.allclose(len(fun3), 2))
    assert(np.allclose(fun3[0], fun4[0]))
    assert(np.allclose(fun3[1], fun4[1]))
    assert(np.allclose(fun4[0], fun4_a))
    assert(np.allclose(fun4[1], fun4_b))

def test_fun2par_single_sample(product_geometry):
    """Test fun2par for single sample"""
    geometry1 = product_geometry.geometries[0]
    geometry2 = product_geometry.geometries[1]

    par2 = (np.ones(geometry1.par_dim), np.ones(geometry2.par_dim)*2)
    fun2 = product_geometry.par2fun(par2[0], par2[1])
    par_same_as_par2 = product_geometry.fun2par(fun2[0], fun2[1])

    assert np.allclose(len(par2), len(par_same_as_par2))
    assert np.allclose(len(par2), 2)
    assert np.allclose(par2[0], par_same_as_par2[0])
    assert np.allclose(par2[1], par_same_as_par2[1])

def test_fun2par_single_sample_stacked(product_geometry):
    """Test fun2par for single sample stacked"""
    geometry1 = product_geometry.geometries[0]
    geometry2 = product_geometry.geometries[1]

    par1 = np.ones(geometry1.par_dim+geometry2.par_dim)
    fun1 = product_geometry.par2fun(par1)
    par_same_as_par1 = product_geometry.fun2par(fun1[0], fun1[1], stacked=True)

    assert np.allclose(par1.shape, par_same_as_par1.shape)
    assert np.allclose(par1, par_same_as_par1)

def test_fun2par_multiple_samples(product_geometry):
    """Test fun2par for multiple samples"""
    geometry1 = product_geometry.geometries[0]
    geometry2 = product_geometry.geometries[1]

    # Number of samples is 5
    Ns = 5

    par4 = (np.ones((geometry1.par_dim, Ns)), 2*np.ones((geometry2.par_dim, Ns)))
    fun4 = product_geometry.par2fun(par4[0], par4[1])
    par_same_as_par4 = product_geometry.fun2par(fun4[0], fun4[1])

    assert np.allclose(len(par4), len(par_same_as_par4))
    assert np.allclose(len(par4), 2)
    assert np.allclose(par4[0], par_same_as_par4[0])
    assert np.allclose(par4[1], par_same_as_par4[1])

def test_fun2par_multiple_samples_stacked(product_geometry):
    """Test fun2par for multiple samples stacked"""
    geometry1 = product_geometry.geometries[0]
    geometry2 = product_geometry.geometries[1]

    # Number of samples is 5
    Ns = 5

    par3 = np.ones((geometry1.par_dim+geometry2.par_dim, Ns))
    par3[geometry1.par_dim:,:] = 2
    fun3 = product_geometry.par2fun(par3)

    par_same_as_par3 = product_geometry.fun2par(fun3[0], fun3[1], stacked=True)

    assert np.allclose(par3.shape, par_same_as_par3.shape)
    assert np.allclose(par3, par_same_as_par3)

def test_number_of_geometries(product_geometry):
    """Test number of geometries"""
    assert np.allclose(product_geometry.number_of_geometries, 2)

def test_stacked_par_split_indices(product_geometry):
    """Test stacked_par_split_indices are correct"""
    assert np.allclose(product_geometry.stacked_par_split_indices, [100])