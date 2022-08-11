"""
Time Dependent Linear PDE
=========================

In this example we show how to set up various Time Dependent Linear PDE models.
"""
# %% 
# First we import the modules needed.

import numpy as np
import sys
from sympy import geometry
sys.path.append("..")
from cuqi.pde import TimeDependentLinearPDE
from cuqi.geometry import Continuous1D
from cuqi.model import PDEModel
from cuqi.samples import CUQIarray

# %%
# Model 1: Heat equation with initial condition as the Bayesian parameter
# -----------------------------------------------------------------------

# Prepare PDE form
dim = 200   # Number of solution nodes
dim_obs = dim
L = 5
max_time = 5
dx = L/(dim+1)   # space step size
cfl = 5/11 # the cfl condition to have a stable solution
max_iter = int(max_time/(cfl*dx**2)) # number of time steps
Dxx = (np.diag( -2*np.ones(dim) ) + np.diag(np.ones(dim-1),-1) + np.diag(np.ones(dim-1),1))/dx**2 # FD diffusion operator

# Grids for model
grid_domain = np.linspace(dx, L, dim, endpoint=False)
grid_range  = grid_domain
time_steps = np.linspace(0,max_time,max_iter,endpoint=True)


# PDE form (diff_op, IC, time_steps)
grid_obs = np.linspace(dx, L, dim_obs, endpoint=False)
def PDE_form(IC, t, dt): return (Dxx, np.zeros(dim), IC)
PDE = TimeDependentLinearPDE(
    PDE_form, time_steps, grid_sol=grid_domain, grid_obs=grid_obs, method="forward_euler")

# Set up geometries for model
domain_geometry = Continuous1D(grid_domain)
range_geometry = Continuous1D(grid_range)

# Prepare model
model = PDEModel(PDE,range_geometry,domain_geometry)

parameters= CUQIarray(np.ones(domain_geometry.dim),geometry=domain_geometry)
solution_case1 = model.forward(parameters)
solution_case1.plot()


# %%
# Model 2: Same as Model 1 but using Backward Euler method for time stepping
# --------------------------------------------------------------------------

cfl = 10 # the cfl condition to have a stable solution
max_iter = int(max_time/(cfl*dx**2)) # number of time steps
time_steps = np.linspace(0,max_time,max_iter,endpoint=True)

# PDE form (diff_op, IC, time_steps)
grid_obs = np.linspace(dx, L, dim_obs, endpoint=False)
def PDE_form(IC, t, dt): return (Dxx, np.zeros(dim), IC)

PDE = TimeDependentLinearPDE(
    PDE_form, time_steps, grid_sol=grid_domain, grid_obs=grid_obs, method="backward_euler")

# Set up geometries for model
domain_geometry = Continuous1D(grid_domain)
range_geometry = Continuous1D(grid_range)

# Prepare model
model = PDEModel(PDE,range_geometry,domain_geometry)

parameters= CUQIarray(np.ones(domain_geometry.dim),geometry=domain_geometry)
solution_case2 = model.forward(parameters)
solution_case2.plot()




np.linalg.norm(solution_case2-solution_case1)/np.linalg.norm(solution_case1)

# reducing cfl to the implicit case will enhance the relative difference

# %%
# Model 3: Same as Model 2, but using varying time step size
# ----------------------------------------------------------

cfl = 10 # the cfl condition to have a stable solution
iter = int(max_time/(cfl*dx**2)) # number of time steps
 # number of time steps

time_steps1 = np.linspace(0,max_time/2,iter+1,endpoint=True)
time_steps2 = np.linspace(max_time/2,max_time,int(iter/2)+1,endpoint=True)
time_steps = np.hstack((time_steps1[:-1],time_steps2))

# PDE form (diff_op, IC, time_steps)
grid_obs = np.linspace(dx, L, dim_obs, endpoint=False)
def PDE_form(IC, t, dt): return (Dxx, np.zeros(dim), IC)

PDE = TimeDependentLinearPDE(
    PDE_form, time_steps, grid_sol=grid_domain, grid_obs=grid_obs, method="backward_euler")

# Set up geometries for model
domain_geometry = Continuous1D(grid_domain)
range_geometry = Continuous1D(grid_range)

# Prepare model
model = PDEModel(PDE,range_geometry,domain_geometry)

parameters= CUQIarray(np.ones(domain_geometry.dim),geometry=domain_geometry)
solution_case3 = model.forward(parameters)
solution_case3.plot()

# %%
# Model 4: Same as model 2 but the source term is the Bayesian parameter
# ----------------------------------------------------------------------
cfl = 10 # the cfl condition to have a stable solution
max_iter = int(max_time/(cfl*dx**2)) # number of time steps
time_steps = np.linspace(0,max_time,max_iter,endpoint=True)

# PDE form (diff_op, source_term, IC)
grid_obs = np.linspace(dx, L, dim_obs, endpoint=False)
def PDE_form(source_term, t, dt): return (Dxx, source_term, np.ones(dim))

PDE = TimeDependentLinearPDE(
    PDE_form, time_steps, grid_sol=grid_domain, grid_obs=grid_obs, method="backward_euler")

# Set up geometries for model
domain_geometry = Continuous1D(grid_domain)
range_geometry = Continuous1D(grid_range)

# Prepare model
model = PDEModel(PDE,range_geometry,domain_geometry)

# Source term zero
parameters= CUQIarray(np.zeros(domain_geometry.dim),geometry=domain_geometry)
solution_case4_a = model.forward(parameters)
solution_case4_a.plot()


# Source term non-zero
parameters= CUQIarray(np.ones(domain_geometry.dim),geometry=domain_geometry)
solution_case4_b = model.forward(parameters)
solution_case4_b.plot()

# %%
# Model 5: First order wave equation with initial condition as the Bayesian parameter
# -----------------------------------------------------------------------------------
# 
# The model set up is similar to the one presented in https://aquaulb.github.io/book_solving_pde_mooc/solving_pde_mooc/notebooks/04_PartialDifferentialEquations/04_01_Advection.html


# Prepare PDE form
dim = 100   # Number of solution nodes
dim_obs = dim
L = 1
max_time = .2
dx = L/(dim+1)   # space step size
dt = .005
max_iter = int(max_time/dt) # number of time steps
Dx = -(  np.diag(1*np.ones(dim-1),1) -np.diag( np.ones(dim),0) )/dx # FD  operator
Dx[0,:]=0



# Grids for model
grid_domain = np.linspace(dx, L, dim, endpoint=False)
grid_range  = grid_domain
# ......
time_steps = np.linspace(0,max_time,max_iter+1,endpoint=True)





# PDE form (diff_op, IC, time_steps)
grid_obs = np.linspace(dx, L, dim_obs, endpoint=False)
def PDE_form(IC, t, dt): return (Dx, np.zeros(dim), IC)
PDE = TimeDependentLinearPDE(
    PDE_form, time_steps, grid_sol=grid_domain, grid_obs=grid_obs, method="forward_euler")

# Set up geometries for model
domain_geometry = Continuous1D(grid_domain)
range_geometry = Continuous1D(grid_range)

# Prepare model
model = PDEModel(PDE,range_geometry,domain_geometry)

IC_func = lambda x: np.exp(-200*(x-L/4)**2)
IC = IC_func(grid_domain)
parameters= CUQIarray(IC,geometry=domain_geometry)
solution_case1 = model.forward(parameters)
solution_case1.plot()

import matplotlib.pyplot as plt
plt.plot(grid_domain,IC)
# %%
