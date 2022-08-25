import numpy as np
import scipy as sp
import cuqi
import pytest
from cuqi.geometry import Continuous2D, Continuous1D

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

@pytest.mark.parametrize("geomClass",
                         [(cuqi.geometry.Continuous1D),
			  (cuqi.geometry.Continuous2D)])
def test_None_Continuous_geometry(geomClass):
    geom = geomClass()
    assert(    (geom.grid == None)
           and (geom.shape == None)
	   and (geom.dim == None))

@pytest.mark.parametrize("geomClass,grid,expected_grid,expected_shape,expected_dim",
                         [(cuqi.geometry.Continuous1D, (4,), np.array([0,1,2,3]),(4,),4),
			  (cuqi.geometry.Continuous2D, (2,3), (np.array([0,1]), np.array([0,1,2])), (2,3), 6)])
def test_update_Continuous_geometry(geomClass,grid,expected_grid,expected_shape,expected_dim):
    geom = geomClass()
    geom.grid = grid
    assert(np.all(np.hstack(geom.grid) == np.hstack(expected_grid))
           and (geom.shape == expected_shape)
	   and (geom.dim == expected_dim))

@pytest.mark.parametrize("variables,expected_variables,expected_shape,expected_dim",
                         [(3,['v0','v1','v2'],(3,),3),
			  (['a','b'],['a','b'],(2,),2),
			  (1,['v0'],(1,),1),
			  ])
def test_Discrete_geometry(variables,expected_variables,expected_shape,expected_dim):
    geom = cuqi.geometry.Discrete(variables)
    assert(geom.variables == expected_variables
           and (geom.shape == expected_shape)
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
    assert(np.allclose(val, imapped_mapped_val) and geom.shape == mapped_geom.shape)
    
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

@pytest.mark.parametrize("n_steps",[1,2,6,7,9,10,20, 21])
def test_StepExpansion_geometry(n_steps):
    """Check StepExpansion geometry correctness"""
    grid = np.linspace(0,1,20)
    if n_steps > np.size(grid):
	#If n_steps is greater than the number of grid points, StepExpansion will fail
        with pytest.raises(ValueError):
            cuqi.geometry.StepExpansion(grid,n_steps)
    else:
	#Otherwise, assert that the StepExpansion is correct
        geom = cuqi.geometry.StepExpansion(grid,n_steps)
        par = np.random.randn(n_steps)
        geom.plot(par,linestyle = '', marker='.')

        assert np.allclose(par, geom.fun2par(geom.par2fun(par))) \
           and geom.dim == n_steps

@pytest.mark.parametrize("projection, func",[('MiN', np.min),
      ('mAX', np.max),('mean', np.mean)])
def test_stepExpansion_fun2par(projection, func):
    """Check StepExpansion fun2par correctness when different projection methods are used"""

    # Set up geometry and grid
    np.random.seed(0)
    grid = np.linspace(0,10, 100, endpoint=True)
    n_steps = 3
    SE_geom = cuqi.geometry.StepExpansion(grid, n_steps=n_steps, projection=projection)
    
    # Create cuqi array of function values
    qa_f = cuqi.samples.CUQIarray(np.random.rand(len(grid)), is_par=False, geometry = SE_geom)
    
    # Compute projection manually (function value to parameters)
    p = np.empty(n_steps)
    p[0]= func(qa_f[np.where(grid <=grid[-1]/n_steps)])
    p[1]= func(qa_f[np.where((grid[-1]/n_steps < grid)&( grid <=2*grid[-1]/n_steps))])
    p[2]= func(qa_f[np.where(2*grid[-1]/n_steps < grid)])

    # Compare projection with fun2par results
    assert np.allclose(p, qa_f.parameters)
