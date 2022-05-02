# %%
# Load cuqi and other packages
import sys
sys.path.append("../../") 
import cuqi
import numpy as np
import matplotlib.pyplot as plt

#Specifically load the CT library (not loaded by default in cuqi)
from cuqi.astra.model import ParBeam2DModel, CT2D_fanbeam
from cuqi.astra.testproblem import ParBeamCT_2D

%load_ext autoreload
%autoreload 2

#%% Define CT model conveniently with cuqi
model = ParBeam2DModel() #CT model with parallel-beam and default values
#model= CT2D_fanbeam() #CT model with fan-beam and default values

# Extract parameters from model
N   = model.domain_geometry.shape[0]
p,q = model.range_geometry.shape
n   = model.domain_geometry.dim #N*N
m   = model.range_geometry.dim  #p*q

# %% Phantom
# Get exact phantom
x_exact = cuqi.data.shepp_logan(size = N)

# Phantom in cuqi array with geometry
x_exact = cuqi.samples.CUQIarray(x_exact, is_par=True, geometry=model.domain_geometry)

# Plot phantom
plt.figure()
x_exact.plot()
plt.colorbar()
#%% Generate exact data and plot it
b_exact = model.forward(x_exact)
plt.figure()
b_exact.plot()
plt.colorbar()

#%% Plot back projection
plt.figure()
model.adjoint(b_exact).plot()
plt.colorbar()

#%% Define Gaussian prior and data distribution
prior      = cuqi.distribution.GaussianCov(np.zeros(n),0.5, geometry = model.domain_geometry)
data_dist  = cuqi.distribution.GaussianCov(model,0.1, geometry = model.range_geometry)

#%% Generate noisy data using the data distribution from x_exact
data=data_dist(x=x_exact).sample()
plt.figure()
data.plot()
plt.colorbar()

#%% Construct likelihood function
likelihood = data_dist.to_likelihood(data)

#%% Posterior distribution
posterior = cuqi.distribution.Posterior(likelihood, prior)

#%% Sample posterior
sampler = cuqi.sampler.Linear_RTO(posterior)
samples = sampler.sample(500,100)
#%% Plot mean
plt.figure()
samples.plot_mean()
plt.colorbar()
#%% Plot std
plt.figure()
samples.plot_std()
plt.colorbar()

#%% Plot posterior samples
plt.figure()
samples.plot()
plt.colorbar()

#%% High level test problem
BP = ParBeamCT_2D(prior=prior, noise_std=0.01, phantom="grains")

cuqi.config.MAX_DIM_INV = 1000 # Change max dim to a lower number such that the problem will be sampled using LinearRTO
samples_BP = BP.sample_posterior(500)

plt.figure()
samples_BP.plot_mean()
plt.colorbar()

plt.figure()
samples_BP.plot_std()
plt.colorbar()
