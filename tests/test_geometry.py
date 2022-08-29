import numpy as np
import scipy as sp
import cuqi
import pytest
from cuqi.geometry import Continuous2D, Continuous1D

@pytest.mark.parametrize("geomClass,grid,expected_grid,expected_par_shape,expected_fun_shape,expected_dim",
                         [(cuqi.geometry.Continuous1D,(1),np.array([0]),(1,),(1,),1),
			  (cuqi.geometry.Continuous1D,(1,),np.array([0]),(1,),(1,),1),
			  (cuqi.geometry.Continuous1D, 1, np.array([0]),(1,),(1,),1),
			  (cuqi.geometry.Continuous1D, [1,2,3,4],np.array([1,2,3,4]),(4,),(4,),4),
			  (cuqi.geometry.Continuous1D, 5,np.array([0,1,2,3,4]),(5,),(5,),5),
			  (cuqi.geometry.Continuous2D,(1,1),(np.array([0]),np.array([0])),(1,),(1,1),1),
			  (cuqi.geometry.Continuous2D,([1,2,3],1), (np.array([1,2,3]), np.array([0])), (3,),(3,1), 3)
			  ])
def test_Continuous_geometry(geomClass,grid,expected_grid,expected_par_shape, expected_fun_shape,expected_dim):
    geom = geomClass(grid=grid)
    assert(np.all(np.hstack(geom.grid) == np.hstack(expected_grid))
           and (geom.par_shape == expected_par_shape)
	   and (geom.dim == expected_dim)
	   and (geom.fun_shape == expected_fun_shape)
	   and (geom.fun_dim == expected_dim))

@pytest.mark.parametrize("geomClass",
                         [(cuqi.geometry.Continuous1D),
			  (cuqi.geometry.Continuous2D)])
def test_None_Continuous_geometry(geomClass):
    geom = geomClass()
    assert(    (geom.grid == None)
           and (geom.par_shape == None)
           and (geom.dim == None)
           and (geom.fun_shape == None)
	   and (geom.fun_dim == None))

@pytest.mark.parametrize("geomClass,grid,expected_grid,expected_shape,expected_dim",
                         [(cuqi.geometry.Continuous1D, (4,), np.array([0,1,2,3]),(4,),4),
			  (cuqi.geometry.Continuous2D, (2,3), (np.array([0,1]), np.array([0,1,2])), (2,3), 6)])
def test_update_Continuous_geometry(geomClass,grid,expected_grid,expected_shape,expected_dim):
    geom = geomClass()
    geom.grid = grid
    assert(np.all(np.hstack(geom.grid) == np.hstack(expected_grid))
           and (geom.fun_shape == expected_shape)
	   and (geom.fun_dim == expected_dim))

@pytest.mark.parametrize("variables,expected_variables,expected_shape,expected_dim",
                         [(3,['v0','v1','v2'],(3,),3),
			  (['a','b'],['a','b'],(2,),2),
			  (1,['v0'],(1,),1),
			  ])
def test_Discrete_geometry(variables,expected_variables,expected_shape,expected_dim):
    geom = cuqi.geometry.Discrete(variables)
    assert(geom.variables == expected_variables
           and (geom.par_shape == expected_shape)
	   and (geom.dim == expected_dim))

@pytest.mark.parametrize("geom1,geom2,truth_value",
                         [(cuqi.geometry._DefaultGeometry(2),cuqi.geometry.Continuous1D(2), True),
			  (cuqi.geometry.Continuous1D(2),cuqi.geometry.Continuous2D((1,2)), False),
			  (cuqi.geometry._DefaultGeometry(np.array([0,1])),cuqi.geometry.Continuous1D(2), True),
			  (cuqi.geometry.Continuous1D(2),cuqi.geometry._DefaultGeometry(3), False),
			  (cuqi.geometry.Discrete(2),cuqi.geometry.Discrete(["v0","v1"]), True),
			  (cuqi.geometry.Discrete(2),cuqi.geometry.Continuous1D(2), False)])
def test_geometry_equivalence(geom1,geom2,truth_value):
    assert( (geom1==geom2) == truth_value)

# Make sure plotting does not fail at least
@pytest.mark.parametrize("is_par,plot_par",	[(True,False),(True,True),(False,False)])
@pytest.mark.parametrize("geom,val",[
						(cuqi.geometry.Discrete(1),np.pi,),
						(cuqi.geometry.Discrete(1),[np.pi]),
						(cuqi.geometry.Discrete(3),[1,2,3]),
						(cuqi.geometry.Continuous1D(1), np.pi,),
						(cuqi.geometry.Continuous1D(1),[np.pi]),
						(cuqi.geometry.Continuous1D(3),[1,2,3]),
						])
def test_geometry_plot(geom,val,is_par,plot_par):
	geom.plot(val,is_par=is_par,plot_par=plot_par)

@pytest.mark.parametrize("is_par,plot_par",	[(True,False),(True,True),(False,False)])
@pytest.mark.parametrize("geom,lo_val,hi_val",[
						(cuqi.geometry.Discrete(1),np.pi,2*np.pi),
						(cuqi.geometry.Discrete(1),[np.pi],[2*np.pi]),
						(cuqi.geometry.Discrete(3),[1,2,3],[4,5,6]),
						(cuqi.geometry.Continuous1D(1),np.pi,2*np.pi),
						(cuqi.geometry.Continuous1D(1),[np.pi],[2*np.pi]),
						(cuqi.geometry.Continuous1D(3),[1,2,3],[4,5,6]),
						])
def test_geometry_plot(geom,lo_val,hi_val,is_par,plot_par):
	geom.plot_envelope(lo_val,hi_val,is_par=is_par,plot_par=plot_par)

def test_geometry_variables_generator_default():
	g1 = cuqi.geometry._DefaultGeometry(5)
	g2 = cuqi.geometry._DefaultGeometry(5)
	g1.variables #Extract variables (they are generated in g1, but not in g2)
	assert g1==g2 #g2 has no _variables yet, but during check its generated.

def test_geometry_variables_generator_Geometry():
	g1 = cuqi.geometry.Continuous1D(5)
	g2 = cuqi.geometry.Continuous1D(5)
	g1.variables #Extract variables (they are generated in g1, but not in g2)
	assert g1==g2 #g2 has no _variables yet, but during check its generated.


@pytest.mark.parametrize("geom, map, imap",
                         [(cuqi.geometry.Discrete(4), lambda x:x**2, lambda x:np.sqrt(x)),
			  (cuqi.geometry.Continuous1D(15),lambda x:x+12, lambda x:x-12),
			  (cuqi.geometry.Continuous2D((4,5)), lambda x:x**2+1, lambda x:np.sqrt(x-1))
			  ])
def test_mapped_geometry(geom, map, imap):
    np.random.seed(0)

    mapped_geom = cuqi.geometry.MappedGeometry(geom, map, imap)
    val = np.random.rand(mapped_geom.dim)
    mapped_val = mapped_geom.par2fun(val)
    imapped_mapped_val = mapped_geom.fun2par(mapped_val)
    assert(np.allclose(val, imapped_mapped_val) and geom.par_shape == mapped_geom.par_shape)
    
@pytest.mark.parametrize("g1, g2, truth_value",[
						(Continuous2D((128,128)), Continuous2D((128,128)), True),
						(Continuous2D((1,2)), Continuous2D((1,2)), True),
						(Continuous2D((3,2)), Continuous2D((3,2)), True),
						(Continuous2D(), Continuous2D(), True),
						(Continuous1D(5), Continuous1D(5), True),
						(Continuous1D(), Continuous1D(), True),
						(Continuous2D((128,128)), Continuous2D((128,127)), False),
						(Continuous2D((1,2)), Continuous2D((1,3)), False),
						(Continuous2D((3,2)), Continuous2D((3,3)), False),
						(Continuous2D(), Continuous2D((1,2)), False),
						(Continuous2D((1,2)), Continuous2D(), False),
						(Continuous1D(5), Continuous1D(6), False),
						(Continuous1D(), Continuous1D(1), False),
						(Continuous1D(1), Continuous1D(), False),
						])
def test_geometry_equality(g1, g2, truth_value):
	"""Ensure geometry arrays compare correctly"""
	assert (g1==g2) == truth_value
