
#%%
import sys
sys.path.append("..")
import cuqi
import matplotlib.pyplot as plt
import numpy as np

#%% Poisson equation in 1D (mixed Dirichlet and Neumann BCs):
# d**2/(dx)**2 u = sin(x), x in (0, pi/2)
# u(0) = g1, and d/dx u(pi/2) = g2

dim = 100 #Number of nodes
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

#%% Plot the solutions 
plt.plot(grid_sol, sol, linestyle='--', marker='.' , label="Solution")
plt.plot(grid_sol, exact_sol(grid_sol), linestyle='--' , label="Exact solution")
plt.xlabel("x")
plt.ylabel("solution (u)")
plt.title("Solution for Poisson problem with mixed BCs")
plt.legend()