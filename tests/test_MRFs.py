import pytest
from cuqi.distribution import LMRF
from cuqi.geometry import Image2D

@pytest.mark.parametrize("scale_value", [0.1, lambda s: s])
def test_LMRF_should_not_allow_None_or_scalar_geometry(scale_value):
    with pytest.raises(ValueError, match="geometry or dim"):
        LMRF(scale=scale_value)

@pytest.mark.parametrize("physical_dim", [-1, 0, 3])
def test_LMRF_invalid_physical_dim_raises_error(physical_dim):
    with pytest.raises(ValueError, match="Only physical dimension 1 or 2 supported."):
        LMRF(scale=0.2, physical_dim=physical_dim, dim=16)

def test_LMRF_mismatched_physical_dim_and_geometry_raises_error():
    image_2d_geometry = Image2D((8, 8))
    with pytest.raises(ValueError):
        LMRF(scale=0.2, physical_dim=1, geometry=image_2d_geometry)

def test_LMRF_default_geometry_is_replaced_when_physical_dim_is_2():
    lmrf = LMRF(scale=0.2, physical_dim=2, dim=16)
    assert isinstance(lmrf.geometry, Image2D)


