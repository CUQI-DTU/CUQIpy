
#%%
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("../")
import cuqi
 
dim = 30
#%%
model, data, problemInfo = cuqi.fenics.testproblem.FEniCSDiffusion1D.get_components(dim = dim)
# %%
prior = cuqi.distribution.GMRF(np.zeros(model.domain_dim),25,model.domain_dim,1,'zero') 

SNR = 100
sigma = np.linalg.norm(problemInfo.exactData)/SNR 
likelihood = cuqi.distribution.GaussianCov(model, sigma**2*np.eye(model.range_dim))
posterior = cuqi.distribution.Posterior(likelihood, prior, data)
# %%
sampler = cuqi.sampler.pCN(posterior)
samples = sampler.sample_adapt(1000)
# %%
samples.plot_ci(95, plot_par = True, exact = problemInfo.exactSolution)
plt.xticks(np.arange(posterior.dim)[::5]);

# %%
samples.plot([100,300])