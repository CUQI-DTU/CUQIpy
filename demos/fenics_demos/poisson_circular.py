# %%
from turtle import pos
from dolfin import * 
import sys
import numpy as np
sys.path.append("../../")
import cuqi
from mshr import *
import matplotlib.pyplot as plt
class source(UserExpression):
    def eval(self,values,x):
        values[0] = 10*np.exp(-(np.power(x[0]-0.5, 2) + np.power(x[1], 2)))

def u_boundary(x, on_boundary):
    return False

#obs_func = lambda m,u : u.split()[0]
obs_func = None
#------------------ Babak's code for boundary extraction
#boundary_elements = AutoSubDomain(lambda x, on_bnd: on_bnd)
#boundary_indicator = DirichletBC(self.V_space, 2, boundary_elements)

#u = Function(self.V_space)
#boundary_indicator.apply( u.vector() )
#values = u.vector()
#self.bnd_idx = np.argwhere( values==2 ).reshape(-1)
#------------------ 

domain = Circle(Point(0,0),1)
mesh = generate_mesh(domain, 20)

V = FiniteElement("CG", mesh.ufl_cell(), 1)
R = FiniteElement("R", mesh.ufl_cell(), 0)
parameter_space = FunctionSpace(mesh, "CG", 1)
solution_space = FunctionSpace(mesh, V*R)
V_space = FunctionSpace(mesh, V)

FEM_el = parameter_space.ufl_element()
source_term = source(element=FEM_el)

#m_func = Function( parameter_space )
def form(m,u,p):
    u_0 = u[0]
    c_0 = u[1]

    v_0 = p[0]
    d_0 = p[1]

    return m*inner( grad(u_0), grad(v_0) )*dx + c_0*v_0*ds + u_0*d_0*ds - source_term*v_0*dx

bc_func = Expression("1", degree=1)
dirichlet_bc = DirichletBC(solution_space.sub(0), bc_func, u_boundary)


PDE = cuqi.fenics.pde.SteadyStateLinearFEniCSPDE( form, mesh, solution_space, parameter_space,dirichlet_bc, observation_operator=obs_func)

#%%
fenics_continuous_geo = cuqi.fenics.geometry.FEniCSContinuous(parameter_space)
matern_geo = cuqi.fenics.geometry.Matern(fenics_continuous_geo, l = .2, nu = 2, num_terms=128)

heavy_map = lambda x: x
domain_geometry = cuqi.fenics.geometry.MappedGeometry(matern_geo, map = heavy_map)
#domain_geometry = matern_geo 

range_geometry = cuqi.fenics.geometry.FEniCSContinuous(solution_space) 

m_input = cuqi.samples.CUQIarray( np.random.standard_normal(128), geometry= domain_geometry)

#m_input = np.random.standard_normal(128)

PDE.assemble(m_input)
sol, _ = PDE.solve()
observed_sol = PDE.observe(sol)

plot(sol[0])

#%%
model = cuqi.model.PDEModel(PDE,range_geometry,domain_geometry)

#%%
# Create prior
pr_mean = np.zeros(domain_geometry.dim)
prior = cuqi.distribution.GaussianCov(pr_mean, cov=np.eye(domain_geometry.dim), geometry= domain_geometry)


# Exact solution
exactSolution = prior.sample()

# Exact data
b_exact = model.forward(domain_geometry.par2fun(exactSolution),is_par=False)

# %%
# Add noise to data
SNR = 100
sigma = np.linalg.norm(b_exact)/SNR
sigma2 = sigma*sigma # variance of the observation Gaussian noise
data = b_exact + np.random.normal( 0, sigma, b_exact.shape )

# Create likelihood
#likelihood = cuqi.distribution.GaussianCov(model, sigma2*np.eye(range_geometry.dim)).to_likelihood(data)
likelihood = cuqi.distribution.GaussianCov(model, sigma2*np.ones(range_geometry.dim)).to_likelihood(data)

posterior = cuqi.distribution.Posterior(likelihood, prior)

#%% MH Sampler
MHSampler = cuqi.sampler.MetropolisHastings(
    posterior,
    proposal=None,
    scale=None,
    x0=None,
    dim=None,
)

samples = MHSampler.sample_adapt(1000)



#%%
plt.figure()
im = plot(domain_geometry.par2fun(exactSolution), title="exact solution")
plt.colorbar(im)

# %%
prior_samples = prior.sample(5)
ims = prior_samples.plot(title="prior")
plt.colorbar(ims[-1])

# %%
ims = samples.plot([0, 100, 300, 600, 800, 900],title="posterior")
plt.colorbar(ims[-1])

# %%
samples.plot_trace()
samples.plot_autocorrelation(max_lag=300)


# %% 
pCNSampler = cuqi.sampler.pCN(
    posterior,
    scale=None,
    x0=None,
)

samplespCN = pCNSampler.sample_adapt(1000)


#%%
plt.figure()
im = plot(domain_geometry.par2fun(exactSolution), title="exact solution")
plt.colorbar(im)

# %%
prior_samples = prior.sample(5)
ims = prior_samples.plot(title="prior")
plt.colorbar(ims[-1])

# %%
ims = samplespCN.plot([0, 100, 300, 600, 800, 900],title="posterior")
plt.colorbar(ims[-1])

# %%
samplespCN.plot_trace()
samplespCN.plot_autocorrelation(max_lag=300)

# %%
plt.figure()
samples.plot_ci(plot_par = True)
plt.title("Credible interval MH")
# %%
