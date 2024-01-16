import pytest
from cuqi.distribution import LMRF, CMRF, GMRF
from cuqi.geometry import Image2D, Geometry, _DefaultGeometry2D
import numpy as np

class Image3D(Geometry):
    """ Test class for 3 dimensional geometry """
    def __init__(self):
        pass
    @property
    def par_shape(self):
        return (8**3,)
    @property
    def fun_shape(self):
        return (8, 8, 8)
    def _plot(self):
        pass
    

@pytest.mark.parametrize("scale_value", [0.1, lambda s: s])
def test_LMRF_should_not_allow_None_or_scalar_geometry(scale_value):
    with pytest.raises(ValueError, match="supported geometry"):
        LMRF(0, scale=scale_value)

@pytest.mark.parametrize("physical_dim", [-1, 0, 3])
def test_LMRF_invalid_physical_dim_raises_error(physical_dim):
    with pytest.raises(ValueError, match="Only physical dimension 1 or 2 supported."):
        LMRF(0, scale=0.2, geometry=Image3D())

def test_LMRF_mismatched_location_and_geometry_raises_error():
    image_2d_geometry = Image2D((8, 8))
    with pytest.raises(TypeError):
        LMRF(np.zeros(3), scale=0.2, geometry=image_2d_geometry)

def test_LMRF_default_geometry_is_replaced_when_physical_dim_is_2():
    lmrf = LMRF(0, scale=0.2, geometry=(4,4))
    assert isinstance(lmrf.geometry, _DefaultGeometry2D)

@pytest.mark.parametrize("scale_value", [0.1, lambda s: s])
def test_CMRF_should_not_allow_None_or_scalar_geometry(scale_value):
    with pytest.raises(ValueError, match="supported geometry"):
        CMRF(0, scale=scale_value)

@pytest.mark.parametrize("physical_dim", [-1, 0, 3])
def test_CMRF_invalid_physical_dim_raises_error(physical_dim):
    with pytest.raises(ValueError, match="Only physical dimension 1 or 2 supported."):
        CMRF(0, scale=0.2, geometry=Image3D())

def test_CMRF_mismatched_location_and_geometry_raises_error():
    image_2d_geometry = Image2D((8, 8))
    with pytest.raises(TypeError):
        CMRF(np.zeros(3), scale=0.2, geometry=image_2d_geometry)

def test_CMRF_default_geometry_is_replaced_when_physical_dim_is_2():
    cmrf = CMRF(0, scale=0.2, geometry=(4,4))
    assert isinstance(cmrf.geometry, _DefaultGeometry2D)

def test_LMRF_dim_vs_ndarray_location():
    lmrf1 = LMRF(0, scale=0.2, geometry=(4,4))
    lmrf2 = LMRF(np.zeros(16), scale=0.2, geometry=Image2D((4,4)))
    assert lmrf1.dim == lmrf2.dim
    assert lmrf1.logd(np.zeros(16)) == lmrf2.logd(np.zeros(16))

def test_CMRF_dim_vs_ndarray_location():
    cmrf1 = CMRF(0, scale=0.2, geometry=(4,4))
    cmrf2 = CMRF(np.zeros(16), scale=0.2, geometry=Image2D((4,4)))
    assert cmrf1.dim == cmrf2.dim
    assert cmrf1.logd(np.zeros(16)) == cmrf2.logd(np.zeros(16))
    assert np.allclose(cmrf1.gradient(np.zeros(16)),cmrf2.gradient(np.zeros(16)))

# GMRF tests
def test_GMRF_should_not_allow_None_or_scalar_geometry():
    with pytest.raises(ValueError, match="supported geometry"):
        GMRF(0, 0.2)

@pytest.mark.parametrize("physical_dim", [-1, 0, 3])
def test_GMRF_invalid_physical_dim_raises_error(physical_dim):
    with pytest.raises(ValueError, match="Only physical dimension 1 or 2 supported."):
        GMRF(0, 0.2, geometry=Image3D())

def test_GMRF_mismatched_location_and_geometry_raises_error():
    image_2d_geometry = Image2D((8, 8))
    with pytest.raises(TypeError):
        GMRF(np.zeros(3), 0.2, geometry=image_2d_geometry)

def test_GMRF_default_geometry_is_replaced_when_physical_dim_is_2():
    gmrf = GMRF(0, 0.2, geometry=(4,4))
    assert isinstance(gmrf.geometry, _DefaultGeometry2D)

def test_GMRF_dim_vs_ndarray_location():
    gmrf1 = GMRF(0, 0.2, geometry=(4,4))
    gmrf2 = GMRF(np.zeros(16), 0.2, geometry=Image2D((4,4)))
    assert gmrf1.dim == gmrf2.dim
    assert gmrf1.logd(np.zeros(16)) == gmrf2.logd(np.zeros(16))
    assert np.allclose(gmrf1.gradient(np.zeros(16)),gmrf2.gradient(np.zeros(16)))


