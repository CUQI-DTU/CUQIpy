#%% Imports
import sys
sys.path.append("..")
import cuqi
import numpy as np
import matplotlib.pyplot as plt
import scipy


#%% Poisson equation
dim = 50 #Number of nodes
L = 10 # Length of the domain
dx = L/(dim-1) # grid spacing 
grid_sol = np.linspace(dx, L, dim-1, endpoint=False)
grid_obs = grid_sol[::8] # we observe evey 8th node
source = lambda a, x0: a*np.exp( -50 * ( (grid_sol - x0)/L)**2 ) #source term
kappa = np.ones(dim)
kappa[np.where(np.arange(dim)>dim/2)] = 2 #kappa is the diffusivity 

#%% ------------------- 1. Solve using default solver -------------------
# Differential operator
FOFD_operator = cuqi.operator.FirstOrderFiniteDifference(dim-1, bc_type='zero', dx=dx).get_matrix().todense()
diff_operator = FOFD_operator.T @np.diag(kappa) @ FOFD_operator

# PDE form
poisson_form = lambda x: (diff_operator, source(x[0], x[1]))

# Create PDE class
CUQI_pde = cuqi.pde.SteadyStateLinearPDE(poisson_form, grid_sol=grid_sol,
					 grid_obs=grid_obs)

# Assemble PDE
x_exact = np.array([10,3]) # [10,3] are the source term parameters [a, x0]
CUQI_pde.assemble(x_exact)

# Solve PDE and observe the solution
sol_1, info_1 = CUQI_pde.solve()
observed_sol_1 = CUQI_pde.observe(sol_1)
print("Information provided by solver:")
print(info_1)


#%% ------------------- 2. Solve using scipy iterative solver -------------------
# Differential operator
FOFD_operator = cuqi.operator.FirstOrderFiniteDifference(dim-1, bc_type='zero', dx=dx).get_matrix()
diff_operator = FOFD_operator.T @np.diag(kappa) @ FOFD_operator

# PDE form
poisson_form = lambda x: (diff_operator, source(x[0], x[1]))

# Create PDE class (passing solver of choice)
linalg_solve_2 = scipy.sparse.linalg.cg
linalg_solve_kwargs_2 = {"x0":np.ones(len(grid_sol))}

CUQI_pde = cuqi.pde.SteadyStateLinearPDE(poisson_form, grid_sol=grid_sol,
					 grid_obs=grid_obs,
					 linalg_solve=linalg_solve_2,
					 linalg_solve_kwargs=linalg_solve_kwargs_2)

# Assemble PDE
x_exact = np.array([10,3]) # [10,3] are the source term parameters [a, x0]
CUQI_pde.assemble(x_exact)

# Solve PDE and observe the solution
sol_2, info_2 = CUQI_pde.solve()
observed_sol_2 = CUQI_pde.observe(sol_2)
print("Information provided by solver:")
print(info_2)


#%% ------------------- 3. Solve using user defined solver -------------------
# Differential operator
FOFD_operator = cuqi.operator.FirstOrderFiniteDifference(dim-1, bc_type='zero', dx=dx).get_matrix().todense()
diff_operator = FOFD_operator.T @np.diag(kappa) @ FOFD_operator

# PDE form
poisson_form = lambda x: (diff_operator, source(x[0], x[1]))

# Create PDE class (passing solver of choice)
def linalg_solve_3(A, b):
# Solve using invertion: not very good approach. But this is to illustrate any third party solver can be wrapped in similar way.
    return cuqi.utilities.force_ndarray(np.linalg.inv(A)@b).flatten(),\
           "dummy_info_1",\
	   "dummy_info_2",\
	   "dummy_info_3" 

linalg_solve_kwargs_3 = {}

CUQI_pde = cuqi.pde.SteadyStateLinearPDE(poisson_form, grid_sol=grid_sol,
					 grid_obs=grid_obs,
					 linalg_solve=linalg_solve_3,
					 linalg_solve_kwargs=linalg_solve_kwargs_3)

# Assemble PDE
x_exact = np.array([10,3]) # [10,3] are the source term parameters [a, x0]
CUQI_pde.assemble(x_exact)

# Solve PDE and observe the solution
sol_3, info_3 = CUQI_pde.solve()
observed_sol_3 = CUQI_pde.observe(sol_3)
print("Information provided by solver:")
print(info_3)


#%% ------------------- 4. Plot the three solutions -------------------
plt.plot(grid_sol, sol_1, linestyle='--', marker='.', label="Solution")
plt.plot(grid_obs, observed_sol_1, linestyle='', marker='*', label="Observed solution")
plt.xlabel("x")
plt.ylabel("solution (u)")
plt.title("Solution using default solver")
plt.legend()

plt.figure()
plt.plot(grid_sol, sol_2, linestyle='--', marker='.', label="Solution")
plt.plot(grid_obs, observed_sol_2, linestyle='', marker='*', label="Observed solution")
plt.xlabel("x")
plt.ylabel("solution (u)")
plt.title("Solution using scipy cg solver")
plt.legend()

plt.figure()
plt.plot(grid_sol, sol_3, linestyle='--', marker='.', label="Solution")
plt.plot(grid_obs, observed_sol_3, linestyle='', marker='*', label="Observed solution")
plt.xlabel("x")
plt.ylabel("solution (u)")
plt.title("Solution using user defined solver")
plt.legend()