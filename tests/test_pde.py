
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

@pytest.mark.parametrize("solve_kwargs, expected_info",
                         [
                          ({}, None),
                          ({"num_info_vars":2}, (2,)),
                          ({"num_info_vars":3}, (2, np.zeros(10))) 
                         ])
def test_solver_signature(solve_kwargs, expected_info):
    # Create solver function that can have different number of returned values
    # based on argument num_returns. It imitates solvers with different signature.
    def mySolve(*args, num_info_vars=1):
        solution = scipy.linalg.solve(*args)
        dummy_info1 = 2
        dummy_info2 = np.zeros(10)
        if num_info_vars == 1:
            # A solver that only returns a solution
            return solution
        elif num_info_vars == 2:
            # A solver that returns a solution and one info variable
            return solution, dummy_info1
        elif num_info_vars == 3:
            # A solver that returns a solution and two info variables
            return solution, dummy_info1, dummy_info2

    # Poisson equation
    dim = 50 #Number of nodes
    L = 10 # Length of the domain
    dx = L/(dim-1) # grid spacing 
    grid_sol = np.linspace(dx, L, dim-1, endpoint=False)
    grid_obs = grid_sol
    source = lambda a, x0: a*np.exp( -50 * ( (grid_sol - x0)/L)**2 ) #source term
    kappa = np.ones(dim)
    kappa[np.where(np.arange(dim)>dim/2)] = 2 #kappa is the diffusivity 

    # Differential operator
    FOFD_operator = cuqi.operator.FirstOrderFiniteDifference(dim-1,bc_type='zero',dx=dx).get_matrix().todense()
    diff_operator = FOFD_operator.T @np.diag(kappa) @ FOFD_operator

    # PDE form
    poisson_form = lambda x: (diff_operator, source(x[0],x[1]))

    # create PDE class
    CUQI_pde = cuqi.pde.SteadyStateLinearPDE(poisson_form, grid_sol=grid_sol, grid_obs=grid_obs, linalg_solve=mySolve, linalg_solve_kwargs=solve_kwargs)

    # assemble PDE
    x_exact = np.array([10,3]) # [10,3] are the source term parameters [a, x0]
    CUQI_pde.assemble(x_exact)

    #solve PDE
    sol, info = CUQI_pde.solve()
    
    # assert first returned value is a numpy array
    assert isinstance(sol, np.ndarray) 
    # assert either info is None (the case when the solver only returns the solution)
    # or each element of info equals to the corresponding element of expected_info (the case in which the solver returned one or more info variables) 
    assert (expected_info == None and info == None) \
               or np.all( [np.all(expected_info[i] == info[i]) for i in range(len(info))]) 






    
