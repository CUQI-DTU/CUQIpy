# %% Initialize and import CUQI
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("..")
import cuqi

# %% Create CUQI test problem
test = cuqi.testproblem._Deblur()
n = test.model.domain_dim

# Extract data
data = test.data

# Extract Likelihood
likelihood = test.likelihood

# Define Prior
loc = np.zeros(n)
prior = cuqi.distribution.CMRF(loc, .2, 'neumann')

# %% Create the posterior and the sampler
posterior = cuqi.distribution.Posterior(likelihood, prior)
MCMC = cuqi.sampler.MALA(posterior, scale=.007, x0=np.ones(n))

# %% Sample
samples = MCMC.sample(3000, 500)

# %% Compute mean
x_mean = np.mean(samples.samples, axis=1)

# %% Compare mean and exact solution by plotting
plt.plot(x_mean)
plt.plot(test.exactSolution)

# %% Plot credibility interval
samples.plot_ci(95, exact=test.exactSolution)

# %% Plot autocorrelation
samples.plot_autocorrelation(max_lag=3000)
