# %%
from dolfin import * 
import sys
import numpy as np
sys.path.append("../../")
import cuqi

source = Expression("sin(x[0])*cos(x[0])", degree =2)
def form(m,u,p):
    u_0 = u[0]
    c_0 = u[1]

    v_0 = p[0]
    d_0 = p[1]
    return m*inner( grad(u_0), grad(v_0) )*dx + c_0*v_0*ds + u_0*d_0*ds - source*v_0*dx

def u_boundary(x, on_boundary):
    return False




obs_func = lambda m,u : u.split()[0]

mesh = UnitSquareMesh(20,20)

V = FiniteElement("CG", mesh.ufl_cell(), 1)
R = FiniteElement("R", mesh.ufl_cell(), 0)
parameter_space = FunctionSpace(mesh, "CG", 1)
solution_space = FunctionSpace(mesh, V*R)
V_space = FunctionSpace(mesh, V)

bc_func = Expression("1", degree=1)
dirichlet_bc = DirichletBC(solution_space.sub(0), bc_func, u_boundary)

m_fun = interpolate(Expression("1", degree=1), parameter_space)

PDE = cuqi.fenics.pde.SteadyStateLinearFEniCSPDE( form, mesh, solution_space, parameter_space,dirichlet_bc, observation_operator=obs_func)

# %%
PDE.assemble(m_fun)
sol = PDE.solve()
observed_sol = PDE.observe(sol)



# %%
#sol_fun = Function(V_space)
#sol_fun.vector().set_local(observed_sol)
# %%
