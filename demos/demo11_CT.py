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
from cuqi.astra.model import CT2D_parallel, CT2D_fanbeam
from cuqi.astra.testproblem import ParBeamCT_2D

%load_ext autoreload
%autoreload 2
# %%
# Define CT model conveniently with cuqi
model = CT2D_parallel() #CT model with parallel-beam and default values
#model= CT2D_fanbeam() #CT model with fan-beam and default values

# %%
# Extract parameters from model
N   = model.domain_geometry.shape[0]
p,q = model.range_geometry.shape
n   = model.domain_geometry.dim #N*N
m   = model.range_geometry.dim  #p*q

# Get exact phantom
phantom = io.loadmat("data/phantom512.mat")["X"]

#Resize phantom 
x_exact_f = interp2d(np.linspace(0,1,phantom.shape[0]),np.linspace(0,1,phantom.shape[1]), phantom)
x_exact = x_exact_f(np.linspace(0,1,N),np.linspace(0,1,N))

# Phantom in cuqi array with geometry
x_exact = cuqi.samples.CUQIarray(x_exact, is_par=True, geometry=model.domain_geometry)

# Plot phantom
plt.figure()
x_exact.plot()
plt.title("Phantom")
# %%
# Generate exact data and plot it
b_exact = model.forward(x_exact)
b_exact.plot()
plt.title("Sinogram")

# %%
# Plot back projection
model.adjoint(b_exact).plot()
plt.title("Back projection")

#%%
# Define Gaussian prior and data distribution
prior      = cuqi.distribution.GaussianCov(np.zeros(n),0.5, geometry = model.domain_geometry)
data_dist  = cuqi.distribution.GaussianCov(model,0.1, geometry = model.range_geometry)

#%%
# Generate noisy data using the data distribution from x_exact
data=data_dist(x=x_exact).sample()
data.plot()
plt.title("noisy sinogram")

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


#%% High level test problem
BP = ParBeamCT_2D(prior=prior,
        data=data, 
        noise_cov=0.1)

samples_BP = BP._sampleLinearRTO(500) # sample_posterior uses _sampleMapCholesky. That is too slow for this problem.

samples_BP.plot_mean(); plt.colorbar()
plt.figure()
samples_BP.plot_std(); plt.colorbar()

