from dolfin import * 
import sys
import numpy as np
sys.path.append("../../")
import cuqi

def form(m,u,p):
    u_0 = u[0]
    c_0 = u[1]

    v_0 = p[0]
    d_0 = p[1]
    return m*inner( grad(u_0), grad(v_0) )*dx + c_0*v_0*ds + u_0*d_0*ds

def u_boundary(x, on_boundary):
    return False




obs_func = lambda m,u : u.split()[0]


mesh = UnitSquareMesh(20,20)

V = FiniteElement("CG", mesh.ufl_cell(), 1)
R = FiniteElement("R", mesh.ufl_cell(), 0)
parameter_space = FunctionSpace(mesh, "CG", 1)
solution_space = FunctionSpace(mesh, V*R)

bc_func = Expression("1", degree=1)
dirichlet_bc = DirichletBC(solution_space.sub(0), bc_func, u_boundary)



PDE = cuqi.fenics.pde.SteadyStateLinearFEniCSPDE( form, mesh, solution_space, parameter_space,dirichlet_bc, observation_operator=obs_func)


