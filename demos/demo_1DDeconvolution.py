#%%
# Import cuqi (assumes cuqipy folder is the same root folder as the sandbox)
# - cuqisandbox
#   - 1DDeconvolution
#     - demo.py
# - cuqipy
#   - cuqi
import sys
sys.path.append("..") 
import cuqi
import matplotlib.pyplot as plt
import numpy as np
import time

# Make sure cuqi is reloaded every time
%load_ext autoreload
%autoreload 2
# %% Set up Deconvolution test problem
tp = cuqi.testproblem.Deconvolution() #Default values

# %% Plot exact solution
plt.plot(tp.exactSolution)
# %% Set a prior and use MAP estimate.
n = tp.model.dim[1]
tp.prior = cuqi.distribution.Gaussian(np.zeros(n),0.1,np.eye(n))
x_map = tp.MAP()
plt.plot(x_map)
plt.plot(tp.exactSolution)

# %% Sample result = tp.sample(Ns)
# Number of samples
Ns = 2000

# Set up target and proposal
def target(x): return tp.likelihood.logpdf(tp.data,x)

# Parameters
scale = 0.02
x0 = np.zeros(n)

# Define sampler
MCMC = cuqi.sampler.pCN(tp.prior,target,scale,x0)

# Run sampler
ti = time.time()
result, target_eval, acc = MCMC.sample(Ns,0)
print('Elapsed time:', time.time() - ti)

# %% plot mean + 95 ci of samples
x_mean = np.mean(result,axis=1)
x_lo95, x_up95 = np.percentile(result, [2.5, 97.5], axis=1)

plt.plot(x_mean)
plt.plot(tp.exactSolution)
plt.fill_between(np.arange(tp.model.dim[1]),x_up95, x_lo95, color='dodgerblue', alpha=0.25)
# %%
