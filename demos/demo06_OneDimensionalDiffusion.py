
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("../")
import cuqi

#%% Set up and solve cuqi problem that uses FEniCSPDEModel 

#%% Problem of form b = A(x) + e
# Create operator A
A = cuqi.PDEmodel.FEniCSDiffusion1D(measurement_type = 'potential')

#%% Create & plot prior 
n = A.dim
pr_mean = np.zeros(n)
prior = cuqi.distribution.GMRF(pr_mean,25,n,1,'zero') 
ps = cuqi.samples.Samples(prior.sample(100))
plt.figure()
ps.plot_ci(95,exact=np.zeros(n))

#%% Create noise (e) and data (b) 
true_m = prior.sample(1)
true_u = A.forward(true_m)
noise_std = 0.01 * np.max(true_u)
e = cuqi.distribution.Gaussian(np.zeros(n), noise_std, np.eye(n))
b = true_u + e.sample().T 

#%% Create cuqi problem (sets up likelihood and posterior) 
IP = cuqi.problem.Type1(b,A,e,prior)

#%% Sample & plot posterior
results = IP.sample(Ns=10000) 
plt.figure()
results.plot_ci(95,exact=true_m)
plt.show()