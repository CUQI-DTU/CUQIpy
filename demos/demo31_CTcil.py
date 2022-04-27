# %%
# Load cuqi and other packages
import sys
sys.path.append("..") 
import cuqi
import numpy as np
import matplotlib.pyplot as plt

#Specifically load the CT library (not loaded by default in cuqi)
from cuqi.cil.model import cilBase, CT2D_parallel, CT2D_fanbeam
from cuqi.cil.testproblem import ParBeamCT_2D

%load_ext autoreload
%autoreload 2

#%% Define CT model conveniently with cuqi
model = CT2D_parallel() #CT model with parallel-beam and default values
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
x_exact.plot(title = "Phantom", origin = "upper-left")
#%% Generate exact data and plot it
b_exact = model.forward(x_exact)
plt.figure()
b_exact.plot(title = "Exact Sinogram", origin = "upper-left")

#%% Plot back projection
b_exact = model.forward(x_exact)
plt.figure()
model.adjoint(b_exact).plot(title = "Back projection", origin = "upper-left")

#%% Define Gaussian prior and data distribution
prior      = cuqi.distribution.GaussianCov(np.zeros(n),0.5, geometry = model.domain_geometry)
data_dist  = cuqi.distribution.GaussianCov(model,0.1, geometry = model.range_geometry)

#%% Generate noisy data using the data distribution from x_exact
data=data_dist(x=x_exact).sample()
plt.figure()
data.plot(title = "Noisy Sinogram", origin = "upper-left")

#%% Construct likelihood function
likelihood = data_dist.to_likelihood(data)

#%% Posterior distribution
posterior = cuqi.distribution.Posterior(likelihood, prior)

#%% Sample posterior
sampler = cuqi.sampler.Linear_RTO(posterior)
samples = sampler.sample(500,100)
#%% Plot mean
plt.figure()
samples.plot_mean(title = "Posterior mean", origin = "upper-left")

#%% Plot std
plt.figure()
samples.plot_std(title = "Posterior std", origin = "upper-left")


#%% High level test problem
BP = ParBeamCT_2D(prior=prior,
        data=data, 
        noise_cov=0.1)

cuqi.config.MAX_DIM_INV = 1000 # Change max dim to a lower number such that the problem will be sampled using LinearRTO
samples_BP = BP.sample_posterior(500)

plt.figure()
samples_BP.plot_mean(title = "Posterior mean", origin = "upper-left")
plt.figure()
samples_BP.plot_std(title = "Posterior std", origin = "upper-left")


# %%
