
#%%
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("../")
import cuqi
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

#%% Set up and solve cuqi problem that uses FEniCSPDEModel 

# Problem of form b = A(x) + e
# Create operator A
model = cuqi.fenics.model.FEniCSDiffusion2D(measurement_type ='sigma_norm_gradu') 

#%% Create & plot prior 
n = model.domain_dim
pr_mean = np.zeros(n)
prior = cuqi.distribution.GMRF(pr_mean,25,n,1,'zero') 
ps = prior.sample(100)
plt.figure()
ps.plot_ci(95,exact=np.zeros(n))
plt.savefig('fig_prior.png')

#%% Create noise (e) and data (b) 
true_m = prior.sample(1)
true_u = model.forward(true_m)
noise_std = float(0.01 * np.max(true_u))
likelihood = cuqi.distribution.Gaussian(model, noise_std, np.eye(n))
b = likelihood(x = true_m).sample()

#%% Create cuqi problem (sets up likelihood and posterior) 

posterior = cuqi.distribution.Posterior(likelihood, prior, b)

#%% Sample & plot posterior
sampler = cuqi.sampler.pCN(posterior)
samples = sampler.sample_adapt(1000)
# %%
samples.plot_ci(95, plot_par = True, exact = true_m, linestyle='-', marker= '')
plt.xticks(ticks=np.arange(posterior.dim)[::200], labels = np.arange(posterior.dim)[::200]);

# %%
samples.plot()
