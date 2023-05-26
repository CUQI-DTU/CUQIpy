# %% Initialize and import CUQI
import sys
sys.path.append("..") 
import numpy as np
import cuqi
import matplotlib.pyplot as plt

# %%
test = cuqi.testproblem._Deblur()
n = test.model.domain_dim
h = test.meshsize

# Extract data
data = test.data

#Likelihood
likelihood  = test.likelihood

# Prior
loc = np.zeros(n)
delta = 1
scale = delta*h
prior = cuqi.distribution.CMRF(loc, scale, 'neumann')

# %%
x0 = cuqi.distribution.Gaussian(np.zeros(n),1).sample()
posterior = cuqi.distribution.Posterior(likelihood, prior)
MCMC = cuqi.sampler.NUTS(posterior,x0, max_depth = 12)

# %%
samples = MCMC.sample(1000,500)

# %%
x_mean = np.mean(samples.samples,axis=1)
# %%
plt.plot(x_mean)
plt.plot(test.exactSolution)
# %%
samples.plot_ci(95,exact=test.exactSolution)
# %%
