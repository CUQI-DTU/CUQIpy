import numpy as np
import scipy as sp
import cuqi
import pytest

@pytest.mark.parametrize("grid,expected_grid,expected_shape,expected_dim",
                         [((1,1),(np.array([0]),np.array([0])),(1,1),1),
			  (([1,2,3],1), (np.array([1,2,3]), np.array([0])), (3,1), 3)
			  ])
def test_continuous2D_geometry(grid,expected_grid,expected_shape,expected_dim):
    geom = cuqi.geometry.Continuous2D(grid=grid)
    assert(np.all(geom.grid[0] == expected_grid[0])
           and np.all(geom.grid[1] == expected_grid[1])
           and (geom.shape == expected_shape)
	   and (geom.dim == expected_dim))