# %% Initialize and import CUQI
import sys
sys.path.append("..") 
import numpy as np
import cuqi
import matplotlib.pyplot as plt

# %% Create CUQI test problem
test = cuqi.testproblem._Deblur()
n = test.model.domain_dim
h = test.meshsize

# Extract data
data = test.data

# Extract Likelihood
likelihood  = test.likelihood

# Define Prior
loc = np.zeros(n)
delta = 1
scale = delta*h
prior = cuqi.distribution.CMRF(loc, scale, 'neumann')

# %% Create the posterior and the sampler
posterior = cuqi.distribution.Posterior(likelihood, prior)
MCMC = cuqi.sampler.ULA(posterior, scale=1/n**2)

# %% Sample
samples  = MCMC.sample(1000, 500)

# %% Compute mean
x_mean = np.mean(samples.samples, axis=1)

# %% Compare mean and exact solution by plotting
plt.plot(x_mean)
plt.plot(test.exactSolution)

# %% Plot credibility interval
samples.plot_ci(95, exact=test.exactSolution)

# %% Plot autocorrelation
samples.plot_autocorrelation(max_lag = 300)
