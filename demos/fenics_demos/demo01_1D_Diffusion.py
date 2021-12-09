
#%%
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("../../")
import cuqi
from math import floor
import dolfin  as dl
import ufl
 
np.random.seed(0)
trial_id = 1
if trial_id == 1:
    map = lambda x : ufl.exp(x)
    form = 'default_form'
else:
    map = lambda x : x
    form = 'exp_form'
dim = 30
N= dim + 1

myExactSolution = np.zeros(N)
myExactSolution[:floor(N/3)] = .2
myExactSolution[floor(N/3):floor(2*N/3)] = .8
myExactSolution[floor(2*N/3):] = .5
#%%
SNR = 100
model, data, problemInfo = cuqi.fenics.testproblem.FEniCSDiffusion1D.get_components(dim = dim, exactSolution = myExactSolution, observation_operator='sigma_u' , SNR = SNR, form = form, map = map)

model.range_geometry.plot(data)


# %%
prior = cuqi.distribution.GMRF(np.zeros(model.domain_dim),25,model.domain_dim,1,'zero', geometry= model.domain_geometry) 


sigma = np.linalg.norm(problemInfo.exactData)/SNR 
likelihood = cuqi.distribution.GaussianCov(model, sigma**2*np.eye(model.range_dim))
posterior = cuqi.distribution.Posterior(likelihood, prior, data)
# %%
sampler = cuqi.sampler.pCN(posterior)
samples = sampler.sample_adapt(1000)
# %%
#samples.burnthin(1000, 100)
samples.plot_ci(95, plot_par = True, exact = problemInfo.exactSolution)
#plt.xticks(np.arange(posterior.dim)[::5]);

# %%
samples.plot([10,30])
samples.plot([1])
# %%
