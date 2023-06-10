import pytest
from cuqi.distribution import LMRF, CMRF
from cuqi.geometry import Image2D
import numpy as np

@pytest.mark.parametrize("scale_value", [0.1, lambda s: s])
def test_LMRF_should_not_allow_None_or_scalar_geometry(scale_value):
    with pytest.raises(ValueError, match="geometry or dim"):
        LMRF(0, scale=scale_value)

@pytest.mark.parametrize("physical_dim", [-1, 0, 3])
def test_LMRF_invalid_physical_dim_raises_error(physical_dim):
    with pytest.raises(ValueError, match="Only physical dimension 1 or 2 supported."):
        LMRF(0, scale=0.2, physical_dim=physical_dim, dim=16)

def test_LMRF_mismatched_physical_dim_and_geometry_raises_error():
    image_2d_geometry = Image2D((8, 8))
    with pytest.raises(ValueError):
        LMRF(0, scale=0.2, physical_dim=1, geometry=image_2d_geometry)

def test_LMRF_default_geometry_is_replaced_when_physical_dim_is_2():
    lmrf = LMRF(0, scale=0.2, physical_dim=2, dim=16)
    assert isinstance(lmrf.geometry, Image2D)

@pytest.mark.parametrize("scale_value", [0.1, lambda s: s])
def test_CMRF_should_not_allow_None_or_scalar_geometry(scale_value):
    with pytest.raises(ValueError, match="geometry or dim"):
        CMRF(0, scale=scale_value)

@pytest.mark.parametrize("physical_dim", [-1, 0, 3])
def test_CMRF_invalid_physical_dim_raises_error(physical_dim):
    with pytest.raises(ValueError, match="Only physical dimension 1 or 2 supported."):
        CMRF(0, scale=0.2, physical_dim=physical_dim, dim=16)

def test_CMRF_mismatched_physical_dim_and_geometry_raises_error():
    image_2d_geometry = Image2D((8, 8))
    with pytest.raises(ValueError):
        CMRF(0, scale=0.2, physical_dim=1, geometry=image_2d_geometry)

def test_CMRF_default_geometry_is_replaced_when_physical_dim_is_2():
    cmrf = CMRF(0, scale=0.2, physical_dim=2, dim=16)
    assert isinstance(cmrf.geometry, Image2D)

def test_LMRF_dim_vs_ndarray_location():
    lmrf1 = LMRF(0, scale=0.2, physical_dim=2, dim=16)
    lmrf2 = LMRF(np.zeros(16), scale=0.2, physical_dim=2)
    assert lmrf1.dim == lmrf2.dim
    assert lmrf1.logd(np.zeros(16)) == lmrf2.logd(np.zeros(16))

def test_CMRF_dim_vs_ndarray_location():
    cmrf1 = CMRF(0, scale=0.2, physical_dim=2, dim=16)
    cmrf2 = CMRF(np.zeros(16), scale=0.2, physical_dim=2)
    assert cmrf1.dim == cmrf2.dim
    assert cmrf1.logd(np.zeros(16)) == cmrf2.logd(np.zeros(16))
    assert np.allclose(cmrf1.gradient(np.zeros(16)),cmrf2.gradient(np.zeros(16)))


