# %%
import sys
sys.path.append("..") 
import cuqi
import matplotlib.pyplot as plt
import numpy as np
import time

# Make sure cuqi is reloaded every time
%load_ext autoreload
%autoreload 2

# Set rng
np.random.seed(0)
# %% Set up Deconvolution test problem
# Parameters
dim = 128
kernel = ["Gauss","Sinc","vonMises"]
phantom = ["Gauss","Sinc","vonMises","Square","Hat","Bumps","DerivGauss"]
noise_type = ["Gaussian","xxxGaussian"]
noise_std = 0.1

# Test problem
tp = cuqi.testproblem.Deconvolution(
    dim = dim,
    kernel=kernel[0],
    phantom=phantom[1],
    noise_type=noise_type[0],
    noise_std = noise_std
)

# Plot exact solution
plt.plot(tp.exactSolution,'.-'); plt.title("Exact solution"); plt.show()

# Plot data
plt.plot(tp.data,'.-'); plt.title("Noisy data"); plt.show()

# %% Set a prior and use MAP estimate.
# Prior
prior_std = 0.1
tp.prior = cuqi.distribution.Gaussian(np.zeros(dim),prior_std,np.eye(dim))

# Map estimate
x_map = tp.MAP()

# Plot
plt.plot(tp.exactSolution,'.-')
plt.plot(x_map,'.-')
plt.title("Map estimate")
plt.legend(["Exact","MAP"])
plt.show()

# %% Sample test problem
# Number of samples
Ns = 5000

# Sample
result = tp.sample(Ns)

# plot mean + 95 ci of samples
x_mean = np.mean(result,axis=1)
x_lo95, x_up95 = np.percentile(result, [2.5, 97.5], axis=1)

plt.plot(x_mean,'.-')
plt.plot(tp.exactSolution,'.-')
plt.fill_between(np.arange(tp.model.dim[1]),x_up95, x_lo95, color='dodgerblue', alpha=0.25)
plt.title("Posterior samples")
plt.legend(["Exact","Posterior mean","Confidense interval"])
plt.show()

# %% Try sampling using a specific sampler
# Set up target and proposal
def target(x): return tp.likelihood.logpdf(tp.data,x)
proposal = tp.prior

# Parameters
scale = 0.02
x0 = np.zeros(dim)

# Define sampler
MCMC = cuqi.sampler.pCN(proposal,target,scale,x0)

# Run sampler
ti = time.time()
result2, target_eval, acc = MCMC.sample(Ns,0)
print('Elapsed time:', time.time() - ti)

# %% plot mean + 95 ci of samples using specific sampler
x_mean_2 = np.mean(result2,axis=1)
x_lo95_2, x_up95_2 = np.percentile(result2, [2.5, 97.5], axis=1)

plt.plot(x_mean_2)
plt.plot(tp.exactSolution)
plt.fill_between(np.arange(tp.model.dim[1]),x_up95_2, x_lo95_2, color='dodgerblue', alpha=0.25)
plt.title("Posterior samples using sampler")
plt.legend(["Exact","Posterior mean","Confidense interval"])

# %%
