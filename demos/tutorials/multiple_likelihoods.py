"""
Setting a Bayesian model with multiple likelihoods 
==================================================

In this example we build a PDE-based Bayesian inverse problem where the Bayesian
model has multiple likelihood functions (two different likelihood functions in
this case, but it can be readily extended to more functions) for the same 
Bayesian parameter `theta`, which represents the conductivity parameters in a
1D Poisson problem. Each likelihood is associated with a different model set
up. The models we use here are obtained from the test problem 
:class:`cuqi.testproblem.Poisson1D`. See the class
:class:`cuqi.testproblem.Poisson1D` documentation for more details about the
forward model.
"""
# %% 
# First we import the python libraries needed.
import cuqi
import numpy as np
import matplotlib.pyplot as plt
from math import ceil

# %%
# Choose one of the two cases we study in this demo 
# -------------------------------------------------
#
# We can choose between two cases:
# Choose `set_up = set_ups[0]` for the case where we have two 1D Poisson models
# that differ in the observation operator only. And choose `set_up = set_ups[1]`
# for the case where we have two 1D Poisson models that differ in the source 
# term only. Here we demonstrate the first case, `set_up = set_ups[0]`.

set_ups = ["multi_observation", "multi_source"]
set_up = set_ups[0]

assert set_up == "multi_observation" or set_up == "multi_source",\
    "set_up must be either 'multi_observation' or 'multi_source'"

# %%
# Set up the parameters used in both models
# -----------------------------------------

dim = 50  # Number of the model grid points
endpoint = 1  # The model domain is the interval [0, endpoint]
field_type = "Step"  # The conductivity (or diffusivity) field type.
                     # We choose step function parameterization here.
SNR = 400  # Signal-to-noise ratio
n_steps = 2  # Number of steps in the conductivity (or diffusivity) step field
magnitude = 100 # Magnitude of the source term in the Poisson problem

# Exact solution
x_exact = np.empty(dim)
x_exact[:ceil(dim/2)] = 2
x_exact[ceil(dim/2):] = 3


# %%
# Set up the first model
# ----------------------
#
# We set up the first forward model to have observations at the first half of 
# the domain (or observation everywhere if `set_up = set_ups[1]`). We then plot
# the true conductivity field (the exact solution), the exact data and the noisy
# data.

observation_grid_map1 = None
if set_up == "multi_observation":
	# Observe on the first half of the domain
	observation_grid_map1 = lambda x: x[np.where(x<.5)] 

# The source term signal
source1 = lambda xs: magnitude*np.sin(xs*2*np.pi/endpoint)+magnitude

# Obtain the forward model from the test problem
model1, data1, problemInfo1 = cuqi.testproblem.Poisson1D(dim=dim,
    endpoint=endpoint,
    field_type=field_type,
    field_params={"n_steps": n_steps},
    observation_grid_map=observation_grid_map1,
    exactSolution=x_exact,
    source=source1,
    SNR=SNR).get_components()

# Plot data, exact data and exact solution
plt.figure()
data1.plot(label='data')
problemInfo1.exactData.plot(label='exact data')
problemInfo1.exactSolution.plot(label='exact solution')
plt.legend()

# %%
# Set up the second model
# -----------------------
#
# We set up the second forward model to have observations at the second half of
# the domain (or observation everywhere and and different source term if
# `set_up = set_ups[1]`). We then plot the true conductivity field (the exact
# solution), the exact data and the noisy data.

observation_grid_map2 = None
if set_up == "multi_observation":
        # Observe on the second half of the domain
	observation_grid_map2 = lambda x: x[np.where(x>=.5)]

# The source term signal
if set_up == "multi_source":
	source2 = lambda xs: magnitude*np.sin(2*xs*2*np.pi/endpoint)+magnitude
else:
	source2 = source1

# Obtain the forward model from the test problem
model2, data2, problemInfo2 = cuqi.testproblem.Poisson1D(dim=dim,
    endpoint=endpoint,
    field_type=field_type,
    field_params={"n_steps": n_steps},
    observation_grid_map=observation_grid_map2,
    exactSolution=x_exact,
    source=source2,
    SNR=SNR).get_components()

# Plot data, exact data and exact solution
plt.figure()
data2.plot(label='data2')
problemInfo2.exactData.plot(label='exact data2')
problemInfo2.exactSolution.plot(label='exact solution2')
plt.legend()

# %%
# Create the prior
# ----------------
#
# Create the prior for the Bayesian parameter `theta`, which is the expansion 
# coefficients of the conductivity (or diffusivity) step function. We use a 
# Gaussian prior.

theta = cuqi.distribution.Gaussian(
    mean=np.zeros(model1.domain_dim),
    cov=3,
    geometry=model1.domain_geometry,
)


# %%
# Create the data distributions using the two forward models
# ----------------------------------------------------------

# Estimate the data noise standard deviation
sigma_noise1 = np.linalg.norm(problemInfo1.exactData)/SNR
sigma_noise2 = np.linalg.norm(problemInfo2.exactData)/SNR

# Create the data distributions
y1 = cuqi.distribution.Gaussian(
    mean=model1(theta),
    cov=sigma_noise1**2,
    geometry=model1.range_geometry,
)
y2 = cuqi.distribution.Gaussian(
    mean=model2(theta),
    cov=sigma_noise2**2,
    geometry=model2.range_geometry,
)


# %%
# Formulate the Bayesian inverse problem using the first data distribution (single likelihood)
# ----------------------------------------------------------------------------------------------------------
# We first formulate the Bayesian inverse problem using the first data distribution and analyze the posterior samples.

# %% 
# Create the posterior 
# ~~~~~~~~~~~~~~~~~~~~

z1 = cuqi.distribution.JointDistribution(theta,y1)(y1=data1)

# %%
# We print the joint distribution `z1`:
print(z1)

# %%
# We see that we obtain a :class:`cuqi.distribution.Posterior` object, which
# represents the posterior distribution of the parameters `theta` given the data
# `y1`. The posterior distribution in this case is proportional to the product
# of the likelihood obtained from the first data distribution and the prior.

# %% 
# Sample from the posterior
# ~~~~~~~~~~~~~~~~~~~~~~~~~~

# Sample from the posterior
sampler = cuqi.sampler.MH(z1)
samples = sampler.sample_adapt(8000)

# Plot the credible interval and compute the ESS
samples.burnthin(1000).plot_ci(95, exact=problemInfo1.exactSolution)
samples.compute_ess()

# %%
# Formulate the Bayesian inverse problem using the second data distribution (single likelihood)
# ------------------------------------------------------------------------------------------------------------
# We then formulate the Bayesian inverse problem using the second data distribution and analyze the posterior samples.

# %% 
# Create the posterior 
# ~~~~~~~~~~~~~~~~~~~~

z2 = cuqi.distribution.JointDistribution(theta,y2)(y2=data2)

# %%
# We print the joint distribution `z2`:
print(z2)

# %%
# We see that we obtain a :class:`cuqi.distribution.Posterior` object, which
# represents the posterior distribution of the parameters `theta` given the data
# `y2`. The posterior distribution in this case is proportional to the product
# of the likelihood obtained from the second data distribution and the prior.

# %% 
# Sample from the posterior
# ~~~~~~~~~~~~~~~~~~~~~~~~~~

# Sample from the posterior
sampler = cuqi.sampler.MH(z2)
samples = sampler.sample_adapt(8000)

# Plot the credible interval and compute the ESS
samples.burnthin(1000).plot_ci(95, exact=problemInfo1.exactSolution)
samples.compute_ess()

# %%
# Formulate the Bayesian inverse problem using both data distributions (multiple likelihoods)
# -------------------------------------------------------------------------------------------
# We then formulate the Bayesian inverse problem using both data distributions and analyze the posterior samples.

# %%
# Create the posterior 
# ~~~~~~~~~~~~~~~~~~~~~

z_joint = cuqi.distribution.JointDistribution(theta,y1,y2)(y1=data1, y2=data2)


# %%
# We print the joint distribution `z_joint`:
print(z_joint)

# %%
# We see that in this case we obtain a :class:`MultipleLikelihoodPosterior` 
# object, which represents the posterior distribution of the parameters `theta`
# given the data `y1` and `y2`. The posterior distribution in this case is 
# proportional to the product of the two likelihoods and the prior.

# %%
# Sample from the posterior 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~

# Sample from the posterior
sampler = cuqi.sampler.MH(z_joint)
samples = sampler.sample_adapt(8000)

# Plot the credible interval and compute the ESS
samples.burnthin(1000).plot_ci(95, exact=problemInfo1.exactSolution)
samples.compute_ess()

# %%
# We notice that combining the two data distributions leads to a more certain 
# estimate of the conductivity (using the same number of MCMC iterations).
# This is because including the two different data sets in the inversion is more
# informative than the single data set case. Also, the effective sample size is
# larger than (or comparable to) what is obtained in any of the single data
# distribution case.
