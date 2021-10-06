#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dst, idst
#from mcmc import Random_Walk
import sys
sys.path.append("../..") 
import cuqi
from heatmodel import heat

#%% Set up model
N = 128
model = heat(N=N)

#%% Set up and plot true initial function using grid from model domain geometry
x = model.domain_geometry.grid
true_init = x*np.exp(-2*x)*np.sin(np.pi-x)
model.domain_geometry.plot(true_init)

#%% defining the heat equation as the forward map
#model.time_stepping(true_init)
y_obs = model._advance_time(true_init, makeplot=True) # observation vector

model.domain_geometry.plot(y_obs)

#%%
SNR = 100 # signal to noise ratio
sigma = np.linalg.norm(y_obs)/SNR   # std of the observation Gaussian noise

likelihood = cuqi.distribution.Gaussian(model,sigma,np.eye(N))

#%% Prior
prior = cuqi.distribution.Gaussian(np.zeros((model.domain_dim,)),1)

#%% Set up problem and sample
IP=cuqi.problem.BayesianProblem(likelihood,prior,y_obs)
results=IP.sample_posterior(5000)

#%% Plot mean
x_mean = np.mean(results.samples,axis=-1)

plt.figure()
model.domain_geometry.plot(x_mean); plt.title("Posterior mean of parameters")

#%%
plt.figure()
model.domain_geometry.plot(model.domain_geometry.to_function(x_mean)); plt.title("Posterior mean in function space")
model.domain_geometry.plot(true_init)
# %%
