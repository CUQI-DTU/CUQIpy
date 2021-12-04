
#%%
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("../")
import cuqi
import dolfin as dl

try: 
    import dolfin as dl
    import ufl
except Exception as error:
    warnings.warn(error.msg)


#%% Define methods
def u_boundary(x, on_boundary):
    return on_boundary

def form(u,m,p):
    return ufl.exp(m)*ufl.inner(ufl.grad(u), ufl.grad(p))*ufl.dx 

#%% Create PDE model
dim = 100
mesh = dl.UnitIntervalMesh(dim)
solution_function_space = dl.FunctionSpace(mesh, 'Lagrange', 1)
parameter_function_space = dl.FunctionSpace(mesh, 'Lagrange', 1)
u_bdr = dl.Expression("x[0]", degree=1)
dirichlet_bc = dl.DirichletBC(solution_function_space, u_bdr, u_boundary)
pde = cuqi.fenics.pde.SteadyStateLinearFEniCSPDE( form, mesh, solution_function_space, parameter_function_space, dirichlet_bc,observation_operator=None)

#%% Test the PDE model
pde.assemble(np.sin(np.arange(dim)))
sol = pde.solve()
dl.plot(sol)