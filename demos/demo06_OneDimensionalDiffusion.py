
#%%
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("../")
import cuqi

np.random.seed(0)
#%% Set up and solve cuqi problem that uses FEniCSPDEModel 

#%% Problem of form b = A(x) + e
# Create operator A
model = cuqi.fenics.pde.FEniCSDiffusion1D(measurement_type = 'potential', shape = (20,))

#%% Create & plot prior 
n = model.dim
pr_mean = np.zeros(n)
prior = cuqi.distribution.GMRF(pr_mean,25,n,1,'zero') 
ps = prior.sample(100)
plt.figure()
ps.plot_ci(95,exact=np.zeros(n))

#%% Create noise (e) and data (b) 
true_m = prior.sample(1)
true_u = model.forward(true_m)
noise_std = float(0.01 * np.max(true_u))
likelihood = cuqi.distribution.Gaussian(mean = model, std = noise_std)
data = likelihood(x=true_m).sample() 

#%% Create cuqi problem (sets up likelihood and posterior) 
IP = cuqi.problem.BayesianProblem( likelihood, prior, data) 

#%% Sample & plot posterior
results = IP.sample_posterior(5000) 

#%% plot
plt.figure()
results.plot_ci(95,exact=true_m, plot_par = True)
plt.show()