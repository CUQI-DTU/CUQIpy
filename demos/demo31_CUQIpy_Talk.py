# %%
import sys
sys.path.append("..")
import cuqi
import numpy as np
import matplotlib.pyplot as plt
import arviz as az

from cuqi.distribution import Posterior, Gaussian, Laplace_diff, Cauchy_diff
from cuqi.testproblem import Deconvolution
from cuqi.sampler import NUTS, Linear_RTO, MALA, CWMH
from cuqi.model import LinearModel

%load_ext autoreload
%autoreload 2

n = 100

# %% The 5 line example

# Deterministic model and data
model, data, probInfo = Deconvolution.get_components(dim=n, phantom='sinc')

# Bayesian model
likelihood = Gaussian(mean=model, std=0.05).to_likelihood(data)
prior      = Gaussian(mean=np.zeros(n), std=0.05)
posterior  = Posterior(likelihood, prior)

# Sampling posterior
samples = NUTS(posterior).sample(1000)

# %% Data
data.plot()

# %% Posterior analysis
# Plot credibility interval
samples.plot_ci()
plt.show()

# Credibility with exact solution to compare with
samples.plot_ci(exact=probInfo.exactSolution)
plt.show()

# Chain plot
samples.plot_chain([30,50,60])
plt.show()

# Pair-wise plot
samples.plot_pair([30,50,60], marginals=True)
plt.show()

samples.plot_trace([30,50,60])
plt.show()

# %% Posterior analysis using arviz

# Convert CUQIpy samples to arviz data structure
samples_arviz = samples.to_arviz_inferencedata()

# Use arviz plot posterior
az.plot_posterior(samples_arviz, var_names=["v30","v50"])
plt.show()

# %% More difficult phantom
# Deterministic model and data
model, data, probInfo = Deconvolution.get_components(dim=n, phantom='pc')

# Bayesian model
likelihood = Gaussian(mean=model, std=0.05).to_likelihood(data)
prior      = Gaussian(mean=np.zeros(n), std=0.5)
posterior  = Posterior(likelihood, prior)

# Sampling posterior
samples = Linear_RTO(posterior).sample(5000,100)

# %% Posterior analysis
samples.plot_ci(exact=probInfo.exactSolution)
plt.show()
samples.plot_trace([30,50,60])
plt.show()

# %% More difficult phantom (switch prior)
# Deterministic model and data
model, data, probInfo = Deconvolution.get_components(dim=n, phantom='square')

# Bayesian model
likelihood = Gaussian(mean=model, std=0.05).to_likelihood(data)
prior      = Cauchy_diff(location=np.zeros(n), scale=0.01, bc_type='zero')
posterior  = Posterior(likelihood, prior)

# Sampling posterior
samples = NUTS(posterior).sample(10000)

# %% Posterior analysis
samples.burnthin(1000).plot_ci(exact=probInfo.exactSolution)
plt.show()
samples.burnthin(1000).plot(samples._select_random_indices(5, 9000), alpha=0.5)
probInfo.exactSolution.plot()
plt.show()
samples.plot_trace([30,50,60])
plt.show()
