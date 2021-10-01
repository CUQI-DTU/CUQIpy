#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dst, idst
#from mcmc import Random_Walk
import sys
sys.path.append("../..") 
import cuqi
from heatmodel import heat


#%% defining the heat equation as the forward map
N = 128
model = heat(N=N, field_type="Step")

#%% computing the true initial condition
# alpha = x * exp(-2x) * sin(pi-x)
true_init = model.domain_geometry.to_function(np.array([3,1,2]))
plt.plot(true_init)

model.set_init_cond(true_init)
model.time_stepping()
y_obs = model.advance_with_init_cond(true_init) # observation vector

plt.plot(y_obs)

#%%
SNR = 100 # signal to noise ratio
sigma = np.linalg.norm(y_obs)/SNR
sigma2 = sigma*sigma # variance of the observation Gaussian noise

likelihood = cuqi.distribution.Gaussian(model,sigma,np.eye(N))

#%% Prior
prior = cuqi.distribution.Gaussian(np.zeros((model.domain_dim,)),1)

#%% Set up problem and sample
IP=cuqi.problem.BayesianProblem(likelihood,prior,y_obs)
results=IP.sample_posterior(5000)

#%% Plot mean
x_mean = np.mean(results.samples,axis=-1)

plt.figure()
model.domain_geometry.plot(x_mean); plt.title("Posterior mean")

#%%
plt.figure()
model.domain_geometry.plot(model.domain_geometry.to_function(x_mean)); plt.title("Posterior mean")
plt.plot(true_init)

# %%
