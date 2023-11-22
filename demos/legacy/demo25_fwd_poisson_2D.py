
#%%
import sys
sys.path.append("..")
import cuqi
import matplotlib.pyplot as plt
import numpy as np
try:
    from findiff import FinDiff, BoundaryConditions
except Exception as err:
    print(err)
    raise ImportError("This demo requires installing the python library findiff")

import scipy as sp

#%%----------------------------------------------------------------------------
# PDE_form function is based on 2D findiff example in https://github.com/maroba/findiff. 
# Create a PDE form for a two dimensional Poisson problem with 
# Neumann and Dirichlet BC. 
shape = (100, 100)

def PDE_form(x, shape=shape):
    # x is the Bayesian parameters.

    #--- 1. set up grid, spacing, and meshgrid
    x_axis, y_axis = np.linspace(0, 1, shape[0]), np.linspace(0, 1, shape[1])
    dx, dy = x_axis[1]-x_axis[0], y_axis[1]-y_axis[0]
    X, Y = np.meshgrid(x_axis, y_axis, indexing='ij')

    #--- 2. Create the Laplace operator L_op and the right-hand-side (rhs) f
    L_op = FinDiff(0, dx, 2) + FinDiff(1, dy, 2)
    f = np.zeros(shape)
    
    #--- 3. Set up boundary conditions
    bc = BoundaryConditions(shape)
    bc[1,:] = FinDiff(0, dx, 1), 0  # Neumann BC (at x=0)
    bc[-1,:] = x[0] - 200*Y   # Dirichlet BC (at x=1)
    bc[:, 0] = x[1]   # Dirichlet BC (at y=0)
    bc[1:-1, -1] = FinDiff(1, dy, 1), 0  # Neumann BC (y=1)

    #--- 4. Apply the BCs to the operator L_op and the rhs f
    # This is based on solve function in https://github.com/maroba/findiff/blob/master/findiff/pde.py
    L_op_matrix = sp.sparse.lil_matrix(L_op.matrix(shape))
    f = f.reshape(-1, 1)
    
    nz = list(bc.row_inds())
    
    L_op_matrix[nz, :] = bc.lhs[nz, :]
    f[nz] = np.array(bc.rhs[nz].toarray()).reshape(-1, 1)
    
    L_op_matrix = sp.sparse.csr_matrix(L_op_matrix)

    #--- 5. return the operator L_op and the rhs f 
    return (L_op_matrix,f)

def observation_map(x):
    return x.reshape(shape)[0] # return solution u where x=0
#%% Poisson equation

# Create cuqi PDE class (passing solver of choice)
linalg_solve = sp.sparse.linalg.spsolve
linalg_solve_kwargs = {}

CUQI_pde = cuqi.pde.SteadyStateLinearPDE(PDE_form,
					 linalg_solve=linalg_solve,
					 linalg_solve_kwargs=linalg_solve_kwargs,
                     observation_map=observation_map)

# Assemble PDE
x_exact = np.array([300,300]) # [300,300] are the values x[0], x[1] in the Dirichlet boundary conditions 
CUQI_pde.assemble(x_exact)

# Solve PDE and apply the observation operator
sol, info = CUQI_pde.solve()
observed_sol = CUQI_pde.observe(sol)

# inspect the output
print("Information provided by solver:")
print(info)

#%%
shape = (100,100) 
x_axis, y_axis = np.linspace(0, 1, shape[0]), np.linspace(0, 1, shape[1])
X, Y = np.meshgrid(x_axis, y_axis, indexing='ij')
im = plt.contourf( X,Y, sol.reshape(shape), levels=20)
plt.colorbar(im)

plt.figure()
plt.title("Observed solution at x=0")
plt.plot(Y[0], observed_sol)
plt.xlabel('x')
plt.ylabel('Solution (u)')
