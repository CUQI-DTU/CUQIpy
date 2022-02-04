
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

def test_mixed_BCs():
    #%% Poisson equation in 1D (mixed Dirichlet and Neumann BCs):
    # d**2/(dx)**2 u = sin(x), x in (0, pi/2)
    # u(0) = g1, and d/dx u(pi/2) = g2
    
    dim = 1000 #Number of nodes
    L = np.pi/2 # Length of the domain
    dx = L/(dim-1) # grid spacing 
    grid_sol = np.linspace(dx, L, dim-1, endpoint=False)
    grid_obs = grid_sol 
    kappa = 1 # kappa is the diffusivity
    g1=2 # Dirichlet bc
    g2=3 # Neumann bc
    
    # Differential operator building blocks (divergence and gradient)
    Div = cuqi.operator.FirstOrderFiniteDifference(dim-1,bc_type='zero',dx=dx).get_matrix().todense()
    Grad = -Div.T
    Grad[0,0] = 0
    Grad[-1,-1] = 0
    
    # PDE form. x[0] is g1 (Dirichlet BC) and x[1] is g2 (Neumann BC).
    def poisson_form(x): #= lambda x: (Laplace, source)
    
        Laplace = kappa*Grad @ Div  #Differential operator (Laplace)
        source = np.sin(grid_sol)  #source term
        # Applying boundary conditions
        Laplace[0,0] = -2/(dx**2)
        source[0] = source[0]- x[0]/(dx**2)
        source[-1] = source[-1]- x[1]/dx
        return (Laplace, source)
    
    # Create PDE class
    CUQI_pde = cuqi.pde.SteadyStateLinearPDE(poisson_form, grid_sol=grid_sol,
    					 grid_obs=grid_obs)
    
    # Assemble PDE
    x_exact = np.array([g1,g2]) # [g1,g2] are the Dirichlet and Neumann BCs values, respectively.
    CUQI_pde.assemble(x_exact)
    
    # Solve PDE
    sol, info = CUQI_pde.solve()
    
    # Exact solution for comparison 
    exact_sol = lambda x: -np.sin(x) + g2*x +g1
    
    # Assert relative error is less than 0.001
    assert(np.linalg.norm(exact_sol(grid_sol) - sol)/np.linalg.norm(sol)<.001)





    
