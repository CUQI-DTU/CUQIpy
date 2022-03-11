# %%
from dolfin import * 
import sys
import numpy as np
sys.path.append("../../")
import cuqi
from mshr import *

class matern():
    def __init__(self, path, num_terms=128):
        self.dim = num_terms
        matern_data = np.load(path)
        self.eig_val = matern_data['l'][:num_terms]
        self.eig_vec = matern_data['e'][:,:num_terms]

    def set_levels(self, c_minus=0., c_plus=1.):
        self.c_minus = c_minus
        self.c_plus = c_plus

    def heavy(self, x):
        return self.c_minus*0.5*(1 + np.sign(x)) + self.c_plus*0.5*(1 - np.sign(x))

    def assemble(self, p):
        return self.eig_vec@( self.eig_val*p )

class source(UserExpression):
    def eval(self,values,x):
        values[0] = 10*np.exp(-(np.power(x[0]-0.5, 2) + np.power(x[1], 2)) )

def u_boundary(x, on_boundary):
    return False

obs_func = lambda m,u : u.split()[0]

domain = Circle(Point(0,0),1)
mesh = generate_mesh(domain, 20)

V = FiniteElement("CG", mesh.ufl_cell(), 1)
R = FiniteElement("R", mesh.ufl_cell(), 0)
parameter_space = FunctionSpace(mesh, "CG", 1)
solution_space = FunctionSpace(mesh, V*R)
V_space = FunctionSpace(mesh, V)

FEM_el = parameter_space.ufl_element()
source_term = source(element=FEM_el)

def form(m,u,p):
    u_0 = u[0]
    c_0 = u[1]

    v_0 = p[0]
    d_0 = p[1]
    return m*inner( grad(u_0), grad(v_0) )*dx + c_0*v_0*ds + u_0*d_0*ds - source_term*v_0*dx

bc_func = Expression("1", degree=1)
dirichlet_bc = DirichletBC(solution_space.sub(0), bc_func, u_boundary)

#m_fun = interpolate(Expression("1", degree=1), parameter_space)
matern_field = matern('basis.npz')
params = matern_field.assemble( np.random.standard_normal(128) )

temp = Function( parameter_space )
temp.vector().set_local( params )

file = File('rand_field.pvd')
file << temp 

print('function created')
exit()


PDE = cuqi.fenics.pde.SteadyStateLinearFEniCSPDE( form, mesh, solution_space, parameter_space,dirichlet_bc, observation_operator=obs_func)

# %%
PDE.assemble(m_fun)
sol = PDE.solve()
observed_sol = PDE.observe(sol)

print( sol.vector().get_local().shape )
print(observed_sol)

V_temp = FunctionSpace(mesh, "CG", 1)
temp = Function(V_temp)
temp.vector().set_local( observed_sol[:-1] )

#path = 'solution.pvd'
#file = File(path)
#file << temp
