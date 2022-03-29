#%% Imports
from turtle import pos, update
from dolfin import * 
import sys
import numpy as np
sys.path.append("../../")
import cuqi
from mshr import *
import matplotlib.pyplot as plt

#%% Define observation map 
obs_func = None

#%% Define domain and mesh
domain = Circle(Point(0,0),1)
mesh = generate_mesh(domain, 20)

# Define function spaces 
V = FiniteElement("CG", mesh.ufl_cell(), 1)
R = FiniteElement("R", mesh.ufl_cell(), 0)
parameter_space = FunctionSpace(mesh, "CG", 1)
solution_space = FunctionSpace(mesh, V*R)
V_space = FunctionSpace(mesh, V)

#%% Define sourceterm
class source(UserExpression):
    def eval(self,values,x):
        values[0] = 10*np.exp(-(np.power(x[0]-0.5, 2) + np.power(x[1], 2)))
FEM_el = parameter_space.ufl_element()
source_term = source(element=FEM_el)

#%% Define Poisson problem
def form(m,u,p):
    u_0 = u[0]
    c_0 = u[1]

    v_0 = p[0]
    d_0 = p[1]

    return m*inner( grad(u_0), grad(v_0) )*dx + c_0*v_0*ds + u_0*d_0*ds - source_term*v_0*dx

#%% Define (dummy) Dirichlet BCs
def u_boundary(x, on_boundary):
    return False
bc_func = Expression("1", degree=1)
dirichlet_bc = DirichletBC(solution_space.sub(0), bc_func, u_boundary)

#%% Create CUQI PDE
PDE = cuqi.fenics.pde.SteadyStateLinearFEniCSPDE( form, mesh, solution_space, parameter_space,dirichlet_bc, observation_operator=obs_func)

#%% Create the domain geometry 
fenics_continuous_geo = cuqi.fenics.geometry.FEniCSContinuous(parameter_space)
matern_geo = cuqi.fenics.geometry.Matern(fenics_continuous_geo, l = .2, num_terms=128)

c_minus = 1
c_plus = 10

def heavy_map(func):
    dofs = func.vector().get_local()
    updated_dofs = c_minus*0.5*(1 + np.sign(dofs)) + c_plus*0.5*(1 - np.sign(dofs))
    func.vector().set_local(updated_dofs)
    return func

domain_geometry = cuqi.fenics.geometry.FEniCSMappedGeometry(matern_geo, map = heavy_map)

#%% Create the range geomtry 
range_geometry = cuqi.fenics.geometry.FEniCSContinuous(solution_space) 

#%% Create CUQI model
model = cuqi.model.PDEModel(PDE,range_geometry,domain_geometry)

#%% Create prior
pr_mean = np.zeros(domain_geometry.dim)
prior = cuqi.distribution.GaussianCov(pr_mean, cov=np.eye(domain_geometry.dim), geometry= domain_geometry)

#%% Define the exact solution
exactSolution = prior.sample()

#%% Generate exact data
b_exact = model.forward(domain_geometry.par2fun(exactSolution),is_par=False)

#%% Add noise to data
SNR = 100
sigma = np.linalg.norm(b_exact)/SNR
sigma2 = sigma*sigma # variance of the observation Gaussian noise
data = b_exact + np.random.normal( 0, sigma, b_exact.shape )

#%% Create likelihood
likelihood = cuqi.distribution.GaussianCov(model, sigma2*np.ones(range_geometry.dim)).to_likelihood(data)

#%% Create posterior
posterior = cuqi.distribution.Posterior(likelihood, prior)

#%% Create MH Sampler
MHSampler = cuqi.sampler.MetropolisHastings(
    posterior,
    proposal=None,
    scale=None,
    x0=None,
    dim=None,
)

#%% Sample using the MH Sampler
samplesMH = MHSampler.sample_adapt(1000)


#%% Plot the exact solution
plt.figure()
im = plot(domain_geometry.par2fun(exactSolution), title="exact solution")
plt.colorbar(im)

#%% Plot prior samples
prior_samples = prior.sample(5)
ims = prior_samples.plot(title="prior")
plt.colorbar(ims[-1])

#%% Plot posterior MH samples
ims = samplesMH.plot([0, 100, 300, 600, 800, 900],title="posterior")
plt.colorbar(ims[-1])

#%% Plot trace and autocorrelation (MH)
samplesMH.plot_trace()
samplesMH.plot_autocorrelation(max_lag=300)


#%% Create pCN Sampler 
pCNSampler = cuqi.sampler.pCN(
    posterior,
    scale=None,
    x0=None,
)

#%% Sample using the pCN sampler
samplespCN = pCNSampler.sample_adapt(1000)

#%% Plot posterior pCN samples 
ims = samplespCN.plot([0, 100, 300, 600, 800, 900],title="posterior")
plt.colorbar(ims[-1])

# %% Plot trace and autocorrelation (pCN)
samplespCN.plot_trace()
samplespCN.plot_autocorrelation(max_lag=300)

#%% Plot credible interval (MH)
plt.figure()
samplesMH.plot_ci(plot_par = True, exact=exactSolution)
plt.xticks(range(128)[::20], range(128)[::20])
plt.title("Credible interval MH")
# %%
