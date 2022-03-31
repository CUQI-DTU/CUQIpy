#%% Imports
import dolfin as dl
import sys
import numpy as np
sys.path.append("../../")
import cuqi
import mshr
import matplotlib.pyplot as plt

#%% 1. First, we create a Poisson PDE using FEniCS. We specify the domain,
# the PDE form, the boundary conditions and the source term (the following
# 6 steps).

#%% 1.1 Define domain and mesh
domain = mshr.Circle(dl.Point(0,0),1)
mesh = mshr.generate_mesh(domain, 20)

#%% 1.2 Define function spaces 
V = dl.FiniteElement("CG", mesh.ufl_cell(), 1)
R = dl.FiniteElement("R", mesh.ufl_cell(), 0)
parameter_space = dl.FunctionSpace(mesh, "CG", 1)
solution_space = dl.FunctionSpace(mesh, V*R)
V_space = dl.FunctionSpace(mesh, V)

#%% 1.3 Define sourceterm
class source(dl.UserExpression):
    def eval(self,values,x):
        values[0] = 10*np.exp(-(np.power(x[0]-0.5, 2) + np.power(x[1], 2)))
FEM_el = parameter_space.ufl_element()
source_term = source(element=FEM_el)

#%% 1.4 Define Poisson problem form
def form(m,u,p):
    u_0 = u[0]
    c_0 = u[1]

    v_0 = p[0]
    d_0 = p[1]

    return m*dl.inner( dl.grad(u_0), dl.grad(v_0) )*dl.dx + c_0*v_0*dl.ds + u_0*d_0*dl.ds - source_term*v_0*dl.dx

#%% 1.5 Define (dummy) Dirichlet BCs
def u_boundary(x, on_boundary):
    return False
bc_func = dl.Expression("1", degree=1)
dirichlet_bc = dl.DirichletBC(solution_space.sub(0), bc_func, u_boundary)

#%% 1.6 Define observation map (applied to the solution to generate the 
# observables)
obs_func = None

#%% 2.Second, we define the Bayesian Inverse probelem using CUQI. We 
# create the geometries for the model domain and range, the forward
# model, the prior, the likelihood and the posterior (the following
# 11 steps).


#%% 2.1 Create the domain geometry 

# 2.1.1 The space on which the Bayesian parameters are defined
fenics_continuous_geo = cuqi.fenics.geometry.FEniCSContinuous(parameter_space)

# 2.1.2 The Matern fieled (maps i.i.d normal random vector of dimension `num_terms`
# to Matern field realization on `fenics_continuous_geo`)
matern_geo = cuqi.fenics.geometry.MaternExpansion(fenics_continuous_geo, length_scale = .8, num_terms=128)

# 2.1.3 We create a map `heavy_map` to map the Matern field realization to two levels
# c_minus and c_plus 
c_minus = 1
c_plus = 10
def heavy_map(func):
    dofs = func.vector().get_local()
    updated_dofs = c_minus*0.5*(1 + np.sign(dofs)) + c_plus*0.5*(1 - np.sign(dofs))
    func.vector().set_local(updated_dofs)
    return func

# 2.1.4 Finally, we create the domain geometry which applies the
# map `heavy_map` on Matern realizations.
domain_geometry = cuqi.fenics.geometry.FEniCSMappedGeometry(matern_geo, map = heavy_map)

#%% 2.2 Create the range geomtry 
range_geometry = cuqi.fenics.geometry.FEniCSContinuous(solution_space) 

#%% 2.3 Create CUQI PDE (which encapsulates the FEniCS formulation
# of the PDE)
PDE = cuqi.fenics.pde.SteadyStateLinearFEniCSPDE( form, mesh, solution_space, parameter_space,dirichlet_bc, observation_operator=obs_func)

#%% 2.4 Create CUQI model
model = cuqi.model.PDEModel(PDE,range_geometry,domain_geometry)

#%% 2.5 Create a prior
pr_mean = np.zeros(domain_geometry.dim)
prior = cuqi.distribution.GaussianCov(pr_mean, cov=np.eye(domain_geometry.dim), geometry= domain_geometry)

#%% 2.6 Define the exact solution
exactSolution = prior.sample()

#%% 2.7 Generate exact data 
b_exact = model(exactSolution)

#%% 2.8 Create the data distribution
SNR = 100
sigma = np.linalg.norm(b_exact)/SNR
sigma2 = sigma*sigma # variance of the observation Gaussian noise
data_distribution = cuqi.distribution.GaussianCov(model, sigma2*np.ones(range_geometry.dim), geometry=range_geometry)

#%% 2.9 Generate noisy data
data = data_distribution(x=exactSolution).sample()

#%% 2.10 Create the data distribution and the likelihood
likelihood = data_distribution.to_likelihood(data)

#%% 2.11 Create posterior
posterior = cuqi.distribution.Posterior(likelihood, prior)


#%% 3 Third, we define a pCN sampler, sample, and inspect the prior and the posterior samples. 

#%% 3.1 Plot the exact solution
exactSolution.plot()

#%% 3.2 Plot prior samples
prior_samples = prior.sample(5)
ims = prior_samples.plot(title="prior")
plt.colorbar(ims[-1])


#%% 3.3 Create pCN Sampler 
pCNSampler = cuqi.sampler.pCN(
    posterior,
    scale=None,
    x0=None,
)

#%% 3.4 Sample using the pCN sampler
samplespCN = pCNSampler.sample_adapt(10000)

#%% 3.5 Plot posterior pCN samples 
ims = samplespCN.plot([0, 1000, 3000, 6000, 8000, 9000],title="posterior")
plt.colorbar(ims[-1])

# %% 3.6 Plot trace and autocorrelation (pCN)
samplespCN.plot_trace()
samplespCN.plot_autocorrelation(max_lag=300)

#%% 3.7 Plot credible interval (pCN)
plt.figure()
samplespCN.plot_ci(plot_par = True, exact=exactSolution)
plt.xticks(range(128)[::20], range(128)[::20])
plt.title("Credible interval")
# %%
