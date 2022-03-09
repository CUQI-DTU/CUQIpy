# %%
# Load cuqi and other packages
import sys
sys.path.append("..") 
import cuqi
import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
import scipy.sparse as sps

#Specifically load the CT library (not loaded by default in cuqi)
from cuqi.astra import CT2D_basic, CT2D_shifted

%load_ext autoreload
%autoreload 2
# %%
# Define CT model conveniently with cuqi
model = CT2D_basic() #CT model with default values
#model= CT2D_shifted() #Shifted detector/source CT model with default values

# %%
# Extract parameters from model
N   = model.domain_geometry.shape[0]
p,q = model.range_geometry.shape
n   = model.domain_geometry.dim #N*N
m   = model.range_geometry.dim  #p*q

# Get exact phantom
x_exact = io.loadmat("data/phantom512.mat")["X"]

#Resize phantom and make into vector
x_exact_f = interp2d(np.linspace(0,1,x_exact.shape[0]),np.linspace(0,1,x_exact.shape[1]), x_exact)
x_exact = x_exact_f(np.linspace(0,1,N),np.linspace(0,1,N))
x_exact = x_exact.ravel()
model.domain_geometry.plot(x_exact); plt.title("Phantom")
# %%
# Generate exact data and plot it
b_exact = model@x_exact 
model.range_geometry.plot(b_exact); plt.title("Sinogram")

# %%
# Plot back projection
model.domain_geometry.plot(model.adjoint(b_exact)); plt.title("Back projection");

#%%
# Define Gaussian prior and data distribution
prior      = cuqi.distribution.GaussianCov(np.zeros(n),0.5)
data_dist  = cuqi.distribution.GaussianCov(model,0.1)

#%%
# Generate noisy data using the data distribution from x_exact
data=data_dist(x=x_exact).sample()
model.range_geometry.plot(data); plt.title("noisy sinogram");

#%%
# Construct likelihood function
likelihood = data_dist.to_likelihood(data)

# %%
posterior = cuqi.distribution.Posterior(likelihood, prior)
sampler = cuqi.sampler.Linear_RTO(posterior)
samples = sampler.sample(500,100)
# %%
# Plot mean
samples.plot_mean(); plt.colorbar()

# %%
# Plot std
samples.plot_std(); plt.colorbar()

# %%
# Sample posterior
IP=cuqi.problem.BayesianProblem(likelihood, prior)
results=IP.sample_posterior(500)
#%%
# Plot mean
x_mean = np.mean(results.samples,axis=-1)
model.domain_geometry.plot(x_mean); plt.title("Posterior mean")
#%%
# Plot standard deviation
x_mean = np.std(results.samples,axis=-1)
model.domain_geometry.plot(x_mean); plt.title("Posterior std")
# %%
