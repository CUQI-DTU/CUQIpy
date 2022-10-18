"""
How to set up multiple likelihoods 
==================================

In this example we show how to set up multiple likelihoods for the same Bayesian parameter `x`, where each likelihood is associated with a different model set up. The model we use here is obtained from the test problem :class:`cuqi.testproblem.Poisson_1D`. See the class :class:`cuqi.testproblem.Poisson_1D` documentation for more details about the forward model.
"""
# %% 
# First we import the modules needed.
import sys
sys.path.append("../../")
import cuqi
import numpy as np
import matplotlib.pyplot as plt
from math import ceil

# %%
# Choose one of the two cases we study in this demo 
# -------------------------------------------------
# Choose `set_up = set_ups[0]` for the case where we have two 1D Poisson models that differ in the observation operator only. And choose `set_up = set_ups[1]` for the case where we have two 1D Poisson models that differ in the source term only.

set_ups = ["multi_observation", "multi_source"]
set_up = set_ups[0]

assert set_up == "multi_observation" or set_up == "multi_source", "set_up must be either 'multi_observation' or 'multi_source'"

# %%
# Set up the parameters used in both models
# -----------------------------------------

dim = 50  # Number of the model grid points
endpoint = 1  # The model domain is the interval [0, endpoint]
field_type = "Step"  # The conductivity (or diffusivity) field type
SNR = 500  # Signal-to-noise ratio
n_steps = 2  # Number of steps in the conductivity (or diffusivity) step field

# Exact solution
x_exact = np.empty(dim)
x_exact[:ceil(dim/2)] = 1
x_exact[ceil(dim/2):] = 3


# %%
# Set up the first model
# ----------------------

observation_grid_map1 = None
if set_up == "multi_observation":
	# Observe on the first half of the domain
	observation_grid_map1 = lambda x: x[np.where(x<.5)] 

# The source term signal
source1 = lambda xs: 20*np.sin(xs)+20.1

# Obtain the forward model from the test problem
model1, data1, problemInfo1 = cuqi.testproblem.Poisson_1D.get_components(dim=dim,
                                                                         endpoint=endpoint,
                                                                         field_type=field_type,
                                                                         field_params={
                                                                             'n_steps': n_steps},
                                                                         observation_grid_map=observation_grid_map1,
                                                                         exactSolution=x_exact,
                                                                         SNR=SNR)

# Plot data, exact data and exact solution
plt.figure()
data1.plot(label='data')
problemInfo1.exactData.plot(label='exact data')
problemInfo1.exactSolution.plot(label='exact solution')
plt.legend()

# %%
# Set up the second model
# -----------------------

observation_grid_map2 = None
if set_up == "multi_observation":
        # Observe on the second half of the domain
	observation_grid_map2 = lambda x: x[np.where(x>=.5)]

# The source term signal
if set_up == "multi_source":
	source2 = lambda xs: 20*np.sin(2*xs)+20.1
else:
	source2 = source1

# Obtain the forward model from the test problem
model2, data2, problemInfo2 = cuqi.testproblem.Poisson_1D.get_components(dim=dim,
                                                                         endpoint=endpoint,
                                                                         field_type=field_type,
                                                                         field_params={
                                                                             'n_steps': n_steps},
                                                                         observation_grid_map=observation_grid_map2,
                                                                         exactSolution=x_exact,
                                                                         SNR=SNR)

# Plot data, exact data and exact solution
plt.figure()
data2.plot(label='data2')
problemInfo2.exactData.plot(label='exact data2')
problemInfo2.exactSolution.plot(label='exact solution2')
plt.legend()

# %%
# Create the prior
# ----------------
# Create the prior for the Bayesian parameter `x`, which is the conductivity (or diffusivity) of the medium. We use a Gaussian prior.

x = cuqi.distribution.GaussianCov(np.zeros(
	model1.domain_dim), 3*np.ones(model1.domain_dim), geometry=model1.domain_geometry)

# %%
# Create the likelihoods
# ----------------------

# Estimate the data noise standard deviation
sigma_noise1 = np.linalg.norm(problemInfo1.exactData)/SNR
sigma_noise2 = np.linalg.norm(problemInfo2.exactData)/SNR

# Create the data distributions
y1 = cuqi.distribution.Gaussian(mean=model1, std=sigma_noise1,
                                corrmat=np.eye(model1.range_dim), geometry=model1.range_geometry)
y2 = cuqi.distribution.Gaussian(mean=model2, std=sigma_noise2,
                                corrmat=np.eye(model2.range_dim), geometry=model2.range_geometry)

# %%
# Create the posterior (multiple likelihoods)
# -------------------------------------------

z_joint = cuqi.distribution.JointDistribution(x,y1,y2)(y1=data1, y2=data2)._as_stacked() # _as_stacked() is needed to stack the random variables but it is a temporary hack that will be not needed in the future.

# %%
# Sample from the posterior (multiple likelihoods)
# ------------------------------------------------

# Sample from the posterior
sampler = cuqi.sampler.MetropolisHastings(z_joint)
samples = sampler.sample_adapt(5000)

# Set the samples geometry
samples.geometry=x.geometry # this is will be not needed in the future as samples geometry will be automatically determined.

# Plot the credible interval and compute the ESS
samples.burnthin(1000).plot_ci(95, exact=problemInfo1.exactSolution)
samples.compute_ess()

# %% 
# Create the posterior (single likelihoods, first model)
# ------------------------------------------------------

z1 = cuqi.distribution.JointDistribution(x,y1)(y1=data1)

# %% 
# Sample from the posterior (single likelihoods, first model)
# -----------------------------------------------------------

# Sample from the posterior
sampler = cuqi.sampler.MetropolisHastings(z1)
samples = sampler.sample_adapt(5000)

# Plot the credible interval and compute the ESS
samples.burnthin(1000).plot_ci(95, exact=problemInfo1.exactSolution)
samples.compute_ess()

# %% 
# Create the posterior (single likelihoods, second model)
# -------------------------------------------------------

z2 = cuqi.distribution.JointDistribution(x,y2)(y2=data2)

# %% 
# Sample from the posterior (single likelihoods, second model)
# ------------------------------------------------------------

# Sample from the posterior
sampler = cuqi.sampler.MetropolisHastings(z2)
samples = sampler.sample_adapt(5000)

# Plot the credible interval and compute the ESS
samples.burnthin(1000).plot_ci(95, exact=problemInfo1.exactSolution)
samples.compute_ess()
