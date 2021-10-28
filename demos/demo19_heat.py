#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dst, idst
import sys
sys.path.append("..") 
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

model.domain_geometry.plot(true_init, is_par=False)
plt.title("Initial")

#%% defining the heat equation as the forward map
y_obs = model._advance_time(true_init, makeplot=True) # observation vector

model.domain_geometry.plot(y_obs, is_par=False)
plt.title("Observed")

#%%
SNR = 100 # signal to noise ratio
sigma = np.linalg.norm(y_obs)/SNR   # std of the observation Gaussian noise

#%% Set up the likelihood. Optionally the geometry can be speficied.
likelihood = cuqi.distribution.Gaussian(mean=model, 
                                        std=sigma, 
                                        corrmat=np.eye(N), 
                                        geometry=model.range_geometry)

#%% Set up the prior. Optionally geometry can be set which allows simple plotting of prior samples
prior = cuqi.distribution.Gaussian(mean=np.zeros((model.domain_dim,)), 
                                   std=1,
                                   geometry=model.domain_geometry)

#%% Generate some prior samples and plot single sample and mean
#   This uses the geometry's plot method which converts from parameters to function values.
prior_samples = prior.sample(1000)

plt.figure()
prior_samples.plot(0)

plt.figure()
prior_samples.plot_mean()

#%% Set up problem and sample
IP = cuqi.problem.BayesianProblem(likelihood, prior, y_obs)
results = IP.sample_posterior(5000)

#%% Compute and plot sample mean of parameters by extracting MANUALLY
x_mean = np.mean(results.samples, axis=-1)

plt.figure()
plt.plot(x_mean)
plt.title("Posterior mean of parameters")

#%% Plot sample mean in function space and compare with true initial
plt.figure()
model.domain_geometry.plot(x_mean)
model.domain_geometry.plot(true_init, is_par=False)
plt.legend(["Sample mean","True initial"])
plt.title("Posterior mean in function space")

# %%
# Plot CI in function space
results.plot_ci(95)
# %%
# Plot CI in parameter space
results.plot_ci(95,plot_par=True)

# %% Plot automatically using geometry
plt.figure()
results.plot_mean()
model.domain_geometry.plot(true_par)
plt.legend(["Sample mean","True initial"])
plt.title("Posterior mean in function space")
# %%
results.plot_ci(95,exact=true_par)
