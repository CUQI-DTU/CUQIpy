
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

def test_observe():
    # Poisson equation
    dim = 20 #Number of nodes
    L = 20 # Length of the domain
    dx = L/(dim-1) # grid spacing 
    grid_sol = np.linspace(dx, L, dim-1, endpoint=False)
    grid_obs = grid_sol[5:]
    source =  lambda mag: mag*np.sin(grid_sol) #source term
    kappa = np.ones(dim) #kappa is the diffusivity 

    # Build the solver
    FOFD_operator = cuqi.operator.FirstOrderFiniteDifference(dim-1, bc_type='zero', dx=dx).get_matrix().todense()
    diff_operator = FOFD_operator.T @ np.diag(kappa) @ FOFD_operator
    poisson_form = lambda x: (diff_operator, source(x[0]))
    CUQI_pde = cuqi.pde.SteadyStateLinearPDE(poisson_form, grid_sol=grid_sol, grid_obs=grid_obs, observation_map=lambda u:u**2)
    x_exact = np.array([2]) # [2] is the source term parameter [mag]
    CUQI_pde.assemble(x_exact)
    sol, info = CUQI_pde.solve()
    observed_sol = CUQI_pde.observe(sol)

    expected_observed_sol =  scipy.linalg.solve(diff_operator, source(2))[5:]**2

    assert(np.all(np.isclose(observed_sol, expected_observed_sol)))


@pytest.mark.parametrize("method, time_steps, parametrization, expected_sol",
                         [('forward_euler', 'fixed', 'initial_condition', 'sol1'),
                          ('backward_euler', 'fixed', 'initial_condition', 'sol2'),
                             ('backward_euler', 'varying',
                              'initial_condition', 'sol3'),
                             ('backward_euler', 'fixed', 'source_term1', 'sol4'),
                             ('backward_euler', 'fixed', 'source_term2', 'sol5')])
def test_TimeDependentLinearPDE_heat1D(copy_reference, method, time_steps, parametrization, expected_sol):
    """ Compute the final time solution of a 1D heat equation and
        compare it with previously stored solution (for 5 different set up choices).
    """
    # 1. Time and space parameters
    dim = 200   # Number of solution nodes
    L = 5  # 1D domain length
    max_time = 1  # Final time
    dx = L/(dim+1)   # Space step size

    if method == 'forward_euler':
        cfl = 5/11  # The cfl condition to have a stable solution
        max_iter = int(max_time/(cfl*dx**2))  # Number of time steps
    elif method == 'backward_euler':
        dt_approx = 0.006  # Approximate time step
        max_iter = int(max_time/dt_approx)  # Number of time steps

    if time_steps == 'fixed':
        time_steps = np.linspace(
            0, max_time, max_iter+1, endpoint=True)  # Time steps array
    elif time_steps == 'varying':
        time_steps1 = np.linspace(0, max_time/2, max_iter+1, endpoint=True)
        time_steps2 = np.linspace(max_time/2, max_time,
                                  int(max_iter/2)+1, endpoint=True)
        time_steps = np.hstack(
            (time_steps1[:-1], time_steps2))  # Time steps array

    # 2. Create differential operator
    Dxx = (np.diag(-2*np.ones(dim)) + np.diag(np.ones(dim-1), -1) +
           np.diag(np.ones(dim-1), 1))/dx**2  # Finite difference diffusion operator

    # 3. Create the PDE form and the Bayesian parameters
    if parametrization == 'initial_condition':
        # PDE form function, returns a tuple of (differential operator, source_term, initial_condition)
        def PDE_form(initial_condition, t): return (
            Dxx, np.zeros(dim), initial_condition)
        parameters = np.ones(dim)
    else:
        # PDE form function, returns a tuple of (differential operator, source_term, initial_condition)
        def PDE_form(source_term, t): return (
            Dxx, source_term, np.ones(dim))
        if parametrization == 'source_term1':
            parameters = np.zeros(dim)
        elif parametrization == 'source_term2':
            parameters = np.ones(dim)

    # 4. Create a PDE object
    PDE = cuqi.pde.TimeDependentLinearPDE(
        PDE_form, time_steps, method=method)

    # 5 Solve the PDE
    PDE.assemble(parameters)
    sol, info = PDE.solve()

    # 6 Compare the obtained solution with previously stored solution
    solution_file = copy_reference("data/Heat1D_5solutions.npz")
    expected_sols = np.load(solution_file)
    assert(np.allclose(sol, expected_sols[expected_sol]))

@pytest.mark.xfail(reason="Test fails due to difficult to compare values (1e-6 to 1e-42)")
def test_TimeDependentLinearPDE_wave1D(copy_reference):
    """ Compute the final time solution of a 1D wave equation and
        compare it with previously stored solution.
    """
    # 1. Time and space parameters
    dim = 100   # Number of solution nodes
    L = 1  # 1D domain length
    max_time = .2  # Final time
    dx = L/(dim+1)   # Space step size
    dt_approx = .005  # Approximate time step
    max_iter = int(max_time/dt_approx)  # Number of time steps
    time_steps = np.linspace(0, max_time, max_iter+1,
                             endpoint=True)  # Time steps array

    # 2. Create the PDE form
    # PDE form function, returns a tuple of (differential operator, source_term, initial_condition)
    Dx = -(np.diag(1*np.ones(dim-1), 1) - np.diag(np.ones(dim), 0)) / \
        dx  # FD advection operator
    Dx[0, :] = 0  # Setting boundary conditions

    def PDE_form(initial_condition, t): return (
        Dx, np.zeros(dim), initial_condition)

    # 3. Create a PDE object
    PDE = cuqi.pde.TimeDependentLinearPDE(
        PDE_form, time_steps, method="forward_euler")

    # 4. Create the initial condition
    grid = np.linspace(dx, L, dim, endpoint=True)
    def initial_condition_func(x): return np.exp(-200*(x-L/4)**2)
    initial_condition = initial_condition_func(grid)

    # 5 Solve the PDE
    PDE.assemble(initial_condition)
    sol, info = PDE.solve()

    # 6 Compare the obtained solution with previously stored solution
    solution_file = copy_reference("data/Wave1D_solution.npz")
    expected_sol = np.load(solution_file)
    assert(np.allclose(sol, expected_sol['sol'], rtol=1e-3, atol=1e-6))
