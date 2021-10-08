#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dst, idst
import sys
sys.path.append("../..") 
import cuqi
from heatmodel import heat

#%% Specify the model
N = 128
field_type = "Step"   # "KL" or "Step" 

#%% Create the model
model = heat(N=N, field_type=field_type)

#%% Set up and plot true initial function using grid from model domain geometry
if field_type=="KL":
    x = model.grid
    true_init = x*np.exp(-2*x)*np.sin(np.pi-x)
else:
    true_par = np.array([3,1,2])
    true_init = model.domain_geometry.par2fun(true_par)

model.domain_geometry.plot(true_init, is_fun=True)
plt.title("Initial")

#%% defining the heat equation as the forward map
y_obs = model._advance_time(true_init, makeplot=True) # observation vector

model.domain_geometry.plot(y_obs, is_fun=True)
plt.title("Observed")

#%%
SNR = 100 # signal to noise ratio
sigma = np.linalg.norm(y_obs)/SNR   # std of the observation Gaussian noise

likelihood = cuqi.distribution.Gaussian(model, sigma, np.eye(N))

#%% Prior
prior = cuqi.distribution.Gaussian(np.zeros((model.domain_dim,)), 1)

#%% Set up problem and sample
IP = cuqi.problem.BayesianProblem(likelihood, prior, y_obs)
results = IP.sample_posterior(5000)

#%% Compute and plot sample mean of parameters
x_mean = np.mean(results.samples, axis=-1)

plt.figure()
plt.plot(x_mean)
plt.title("Posterior mean of parameters")

#%% Plot sample mean in function space and compare with true initial
plt.figure()
model.domain_geometry.plot(x_mean)
model.domain_geometry.plot(true_init, is_fun=True)
plt.legend(["Sample mean","True initial"])
plt.title("Posterior mean in function space")

# %%
