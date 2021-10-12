import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dst, idst
import sys
sys.path.append("../..") 
import cuqi
from abelmodel import abel

#%% Specify the model
N = 100

#%% Create the model
model = abel(N=N)

#%% Set up and plot true initial function using grid from model domain geometry
tvec = model.tvec.reshape(-1)
true_im = np.sin(tvec*np.pi)*np.exp(-2*tvec)

model.domain_geometry.plot(true_im, is_fun=True)
plt.title("Initial")

#%% defining the heat equation as the forward map
y_obs = model.solve_with_image(true_im) # observation vector

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
model.domain_geometry.plot(true_im, is_fun=True)
plt.legend(["Sample mean","True initial"])
plt.title("Posterior mean in function space")
# %%