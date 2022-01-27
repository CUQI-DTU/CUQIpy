
import numpy as np
import matplotlib.pyplot as plt
import sys
import cuqi
import pytest
import scipy


def test_solver_choice():
    # Poisson equation
    dim = 50 #Number of nodes
    L = 10 # Length of the domain
    dx = L/(dim-1) # grid spacing 
    grid_sol = np.linspace(dx, L, dim-1, endpoint=False)
    grid_obs = grid_sol
    source = lambda a, x0: a*np.exp( -50 * ( (grid_sol - x0)/L)**2 ) #source term
    kappa = np.ones(dim)
    kappa[np.where(np.arange(dim)>dim/2)] = 2 #kappa is the diffusivity 


    # Solver1 (default)
    FOFD_operator = cuqi.operator.FirstOrderFiniteDifference(dim-1,bc_type='zero',dx=dx).get_matrix().todense()
    diff_operator = FOFD_operator.T @np.diag(kappa) @ FOFD_operator
    poisson_form = lambda x: (diff_operator, source(x[0],x[1]))
    CUQI_pde = cuqi.pde.SteadyStateLinearPDE(poisson_form, grid_sol=grid_sol, grid_obs=grid_obs)
    x_exact = np.array([10,3]) # [10,3] are the source term parameters [a, x0]
    CUQI_pde.assemble(x_exact)
    sol, info = CUQI_pde.solve()
    observed_sol_1 = CUQI_pde.observe(sol)

    #Solver2 (sparse)
    FOFD_operator = cuqi.operator.FirstOrderFiniteDifference(dim-1,bc_type='zero',dx=dx).get_matrix()
    diff_operator = FOFD_operator.T @scipy.sparse.diags(kappa) @ FOFD_operator
    poisson_form = lambda x: (diff_operator, source(x[0],x[1]))
    CUQI_pde = cuqi.pde.SteadyStateLinearPDE(poisson_form, grid_sol=grid_sol, grid_obs=grid_obs, linalg_solve=scipy.sparse.linalg.spsolve, linalg_solve_kwargs={"use_umfpack":False})
    x_exact = np.array([10,3]) # [10,3] are the source term parameters [a, x0]
    CUQI_pde.assemble(x_exact)
    sol, info = CUQI_pde.solve()
    observed_sol_2 = CUQI_pde.observe(sol)

    #Solver3 (cg)
    def mycg(*args,**kwargs):
        x,_ = scipy.sparse.linalg.cg(*args,**kwargs) 
        return x
    CUQI_pde = cuqi.pde.SteadyStateLinearPDE(poisson_form, grid_sol=grid_sol, grid_obs=grid_obs, linalg_solve=mycg)
    x_exact = np.array([10,3]) # [10,3] are the source term parameters [a, x0]
    CUQI_pde.assemble(x_exact)
    sol, info = CUQI_pde.solve()
    observed_sol_3 = CUQI_pde.observe(sol)

    assert(np.allclose(observed_sol_1,observed_sol_2) and np.allclose(observed_sol_1,observed_sol_3))






    
