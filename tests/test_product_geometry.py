# %%
import cuqi
import pytest
import numpy as np
from cuqi.geometry import Continuous1D, Discrete, MappedGeometry, Continuous2D
from cuqi.geometry import _ProductGeometry

@pytest.fixture
def product_geometry():
    """fixture function to create a product geometry"""
    geometry1 = Continuous1D(np.linspace(0, 1, 100))
    geometry2 = MappedGeometry(
        Discrete(["sound_speed"]), map=lambda x: x**2, imap=lambda x: np.sqrt(x)
    )
    geometry3 = Continuous2D((3,3))
    product_geometry = _ProductGeometry(geometry1, geometry2, geometry3)
    return product_geometry

@pytest.fixture
def par1(product_geometry):
    """fixture function to create a parameter vector"""
    geometry1, geometry2, geometry3 = product_geometry.geometries

    g1_dim = geometry1.par_dim
    g2_dim = geometry2.par_dim
    g3_dim = geometry3.par_dim

    par1 = np.ones(g1_dim+g2_dim+g3_dim)

    par1[g1_dim:g1_dim+g2_dim] = 2
    par1[g1_dim+g2_dim:] = 3
    return par1

@pytest.fixture
def par2(product_geometry):
    """fixture function to create a parameter vector"""
    geometry1, geometry2, geometry3 = product_geometry.geometries

    par2 = (np.ones(geometry1.par_dim),
            np.ones(geometry2.par_dim)*2,
            np.ones(geometry3.par_dim)*3)
    return par2

@pytest.fixture
def par3(product_geometry):
    """fixture function to create a parameter vector"""
    geometry1, geometry2, geometry3 = product_geometry.geometries

    g1_dim = geometry1.par_dim
    g2_dim = geometry2.par_dim
    g3_dim = geometry3.par_dim
    
    Ns = 5
    par3 = np.ones((g1_dim+g2_dim+g3_dim, Ns))
    par3[g1_dim:,:] = 2
    par3[g1_dim+g2_dim:,:] = 3

    return par3

@pytest.fixture
def par4(product_geometry):
    """fixture function to create a parameter vector"""
    geometry1, geometry2, geometry3 = product_geometry.geometries

    Ns = 5
    par4 = (np.ones((geometry1.par_dim, Ns)),
            2*np.ones((geometry2.par_dim, Ns)),
            3*np.ones((geometry3.par_dim, Ns)))
    return par4


def test_par2fun_single_sample(product_geometry, par1, par2):
    """Test par2fun for single sample"""
    geometry1, geometry2, geometry3 = product_geometry.geometries

    fun1 = product_geometry.par2fun(par1)
    fun2 = product_geometry.par2fun(par2[0], par2[1], par2[2])
    fun2_a = geometry1.par2fun(par2[0])
    fun2_b = geometry2.par2fun(par2[1])
    fun2_c = geometry3.par2fun(par2[2])

    assert np.allclose(len(fun1), 3)
    assert np.allclose(len(fun1), len(fun2))
    assert np.allclose(fun1[0], fun2[0])
    assert np.allclose(fun1[1], fun2[1])
    assert np.allclose(fun1[2], fun2[2])
    assert np.allclose(fun2[0], fun2_a)
    assert np.allclose(fun2[1], fun2_b)
    assert np.allclose(fun2[2], fun2_c)

def test_par2fun_multiple_samples(product_geometry, par3, par4):
    """Test par2fun for multiple samples"""
    geometry1, geometry2, geometry3 = product_geometry.geometries

    fun3 = product_geometry.par2fun(par3)
    fun4 = product_geometry.par2fun(par4[0], par4[1], par4[2])
    fun4_a = geometry1.par2fun(par4[0])
    fun4_b = geometry2.par2fun(par4[1])
    fun4_c = geometry3.par2fun(par4[2])

    assert(np.allclose(len(fun3), len(fun4)))
    assert(np.allclose(len(fun3), 3))
    assert(np.allclose(fun3[0], fun4[0]))
    assert(np.allclose(fun3[1], fun4[1]))
    assert(np.allclose(fun3[2], fun4[2]))
    assert(np.allclose(fun4[0], fun4_a))
    assert(np.allclose(fun4[1], fun4_b))
    assert(np.allclose(fun4[2], fun4_c))

def test_fun2par_single_sample(product_geometry, par2):
    """Test fun2par for single sample"""

    fun2 = product_geometry.par2fun(par2[0], par2[1], par2[2])
    par_same_as_par2 = product_geometry.fun2par(fun2[0], fun2[1], fun2[2])

    assert np.allclose(len(par2), len(par_same_as_par2))
    assert np.allclose(len(par2), 3)
    assert np.allclose(par2[0], par_same_as_par2[0])
    assert np.allclose(par2[1], par_same_as_par2[1])
    assert np.allclose(par2[2], par_same_as_par2[2])

def test_fun2par_single_sample_stacked(product_geometry, par1):
    """Test fun2par for single sample stacked"""

    fun1 = product_geometry.par2fun(par1)
    par_same_as_par1 = product_geometry.fun2par(fun1[0], fun1[1], fun1[2], stacked=True)

    assert np.allclose(par1.shape, par_same_as_par1.shape)
    assert np.allclose(par1, par_same_as_par1)

def test_fun2par_multiple_samples(product_geometry, par4):
    """Test fun2par for multiple samples"""

    fun4 = product_geometry.par2fun(par4[0], par4[1], par4[2])
    par_same_as_par4 = product_geometry.fun2par(fun4[0], fun4[1], fun4[2])

    assert np.allclose(len(par4), len(par_same_as_par4))
    assert np.allclose(len(par4), 3)
    assert np.allclose(par4[0], par_same_as_par4[0])
    assert np.allclose(par4[1], par_same_as_par4[1])
    assert np.allclose(par4[2], par_same_as_par4[2])

def test_fun2par_multiple_samples_stacked(product_geometry, par3):
    """Test fun2par for multiple samples stacked"""

    fun3 = product_geometry.par2fun(par3)

    par_same_as_par3 = product_geometry.fun2par(fun3[0], fun3[1], fun3[2], stacked=True)

    assert np.allclose(par3.shape, par_same_as_par3.shape)
    assert np.allclose(par3, par_same_as_par3)

def test_number_of_geometries(product_geometry):
    """Test number of geometries"""
    assert np.allclose(product_geometry.number_of_geometries, 3)

def test_stacked_par_split_indices(product_geometry):
    """Test stacked_par_split_indices are correct"""
    assert np.allclose(product_geometry.stacked_par_split_indices, [100, 101])

def test_input_not_geometry_fails():
    """Test that passing input other than Geometry fails"""
    with pytest.raises(TypeError,
                       match="All geometries must be of type Geometry"):
        _ProductGeometry(1, 2, 3)