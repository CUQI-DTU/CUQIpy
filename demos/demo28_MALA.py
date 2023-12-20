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
mala_sampler = cuqi.sampler.MALA(posterior, scale=.007, x0=np.ones(n))

# %% Sample
mala_samples = mala_sampler.sample(9000, 1000)

# %% Compute mean
mala_mean = np.mean(mala_samples.samples, axis=1)

# %% Compare mean and exact solution by plotting
plt.plot(mala_mean)
plt.plot(test.exactSolution)

# %% Plot credibility interval
mala_samples.plot_ci(95, exact=test.exactSolution)

# %% Plot autocorrelation
mala_samples.plot_autocorrelation(max_lag=3000)

# %% MALAL
malal_sampler = cuqi.sampler.MALAL(posterior, scale=.007, x0=np.ones(n), tmax=1, coeff=posterior.dim**(-1/3))

# %% Sample
malal_samples = malal_sampler.sample(9000, 1000)

# %% Compute mean
malal_mean = np.mean(malal_samples.samples, axis=1)

# %% Compare mean and exact solution by plotting
plt.plot(malal_mean)
plt.plot(test.exactSolution)

# %% Plot credibility interval
malal_samples.plot_ci(95, exact=test.exactSolution)

# %% Plot autocorrelation
malal_samples.plot_autocorrelation(max_lag=3000)

