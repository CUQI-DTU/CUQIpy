
#%%
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("../../")
import cuqi

np.random.seed(0)
mapping =  'exponential'
dim = 30
N= dim + 1
L = 1
myExactSolution= 'smooth_step'

#%%
observation_operator=None
SNR = 1000
model, data, problemInfo = cuqi.fenics.testproblem.FEniCSDiffusion1D.get_components(dim = dim, exactSolution = myExactSolution, observation_operator=observation_operator , SNR = SNR, mapping = mapping, left_bc = 1, right_bc = 8, endpoint=L)

model.range_geometry.plot(data)
plt.title('Data')


# %%
prior = cuqi.distribution.GMRF(np.zeros(model.domain_dim),25,1,'zero', geometry= model.domain_geometry) 


sigma = np.linalg.norm(problemInfo.exactData)/SNR 
likelihood = cuqi.distribution.GaussianCov(model, sigma**2*np.eye(model.range_dim)).to_likelihood(data)
posterior = cuqi.distribution.Posterior(likelihood, prior)
# %%
sampler = cuqi.sampler.pCN(posterior)
samples = sampler.sample_adapt(5000)
# %%

samples.plot_ci(95, plot_par = True, exact = problemInfo.exactSolution, linestyle='-', marker='.')
plt.xticks(np.arange(prior.dim)[::5],['v'+str(i) for i in range(prior.dim)][::5]);


# %%
samples.plot([10,24]);
# %%

# %%
new_samples = samples.burnthin(1000) 
new_samples.plot_ci(95, plot_par = True, exact = problemInfo.exactSolution, linestyle='-', marker='.')
plt.xticks(np.arange(prior.dim)[::5],['v'+str(i) for i in range(prior.dim)][::5]);
