import numpy as np
import scipy as sp
import cuqi
import pytest

@pytest.mark.parametrize("geomClass,grid,expected_grid,expected_shape,expected_dim",
                         [(cuqi.geometry.Continuous1D,(1),np.array([0]),(1,),1),
			  (cuqi.geometry.Continuous1D,(1,),np.array([0]),(1,),1),
			  (cuqi.geometry.Continuous1D, 1, np.array([0]),(1,),1),
			  (cuqi.geometry.Continuous1D, [1,2,3,4],np.array([1,2,3,4]),(4,),4),
			  (cuqi.geometry.Continuous1D, 5,np.array([0,1,2,3,4]),(5,),5),
			  (cuqi.geometry.Continuous2D,(1,1),(np.array([0]),np.array([0])),(1,1),1),
			  (cuqi.geometry.Continuous2D,([1,2,3],1), (np.array([1,2,3]), np.array([0])), (3,1), 3)
			  ])
def test_Continuous_geometry(geomClass,grid,expected_grid,expected_shape,expected_dim):
    geom = geomClass(grid=grid)
    assert(np.all(np.hstack(geom.grid) == np.hstack(expected_grid))
           and (geom.shape == expected_shape)
	   and (geom.dim == expected_dim))

