# %%
import sys
import time
sys.path.append("..")

import numpy as np
import matplotlib.pyplot as plt

import cuqi

#%% %Cuqi deblur test problem
tp = cuqi.testproblem._Deblur() #Default values

#%% Plot true, blurred and noisy data
plt.figure
plt.plot(tp.mesh,tp.exactSolution)
plt.plot(tp.mesh,tp.exactData)
plt.plot(tp.mesh,tp.data)
plt.ylim([-0.5,3.5])
plt.legend(['True','Blurred','Blurred and noisy'])

#%% Unpack problem
b = tp.data;        # Measured data
A = tp.model        # Class with model
L = tp.likelihood   # Class with likelihood

#%% A few additional parameters from test problem needed
h = tp.meshsize;        # Size of mesh elements
n = tp.model.domain_dim;    # Number of unknowns

#%% Two choices of prior

P1 = cuqi.distribution.GMRF(np.zeros(n), 25, 'zero', name="x")

loc = np.zeros(n)
delta = 1
scale = delta*h
P2 = cuqi.distribution.CMRF(loc, scale, 'neumann', name="x")

#%% Generate and display some prior samples
sp1 = P1.sample(5)
#sp2 = prior2.sample(5)

plt.figure
plt.subplot(1,2,1)
sp1.plot()
plt.title('GMRF')
#TODO
#subplot(1,2,2)
#sp2.plot(5)
#title('Cauchy')

#%% Number of samples
Ns = 200

#%% 1.  "High level"  - set up cuqi Problem

# Define Bayesian model
IP = cuqi.problem.BayesianProblem(L, P1)

# Then we can simply sample the posterior
results_prior1 = IP.sample_posterior(Ns) 

# Plot 95% credibility interval
results_prior1.plot_ci(95,exact=tp.exactSolution)

#%% 2.  "Absolute non-expert level" -  just ask for UQ!

#The blur TestProblem is a subclass of cuqi.Problem, just need to add prior
tp.prior = P2

#Use UQ convenience method:
results_prior2 = tp.sample_posterior(Ns)

#%%
norm_f = np.linalg.norm(tp.exactSolution)
med_xpos1 = np.median(results_prior1.samples, axis=1) # sp.stats.mode
sigma_xpos1 = results_prior1.samples.std(axis=1)
relerr = round(np.linalg.norm(med_xpos1 - tp.exactSolution)/norm_f*100, 2)
print('\nGMRF Relerror median:', relerr, '\n')

med_xpos2 = np.median(results_prior2.samples, axis=1) # sp.stats.mode
sigma_xpos2 = results_prior2.samples.std(axis=1)
relerr = round(np.linalg.norm(med_xpos2 - tp.exactSolution)/norm_f*100, 2)
print('\nCauchy Relerror median:', relerr, '\n')

#%%
plt.figure()
plt.plot(tp.mesh, tp.exactSolution, 'k-')
plt.plot(tp.mesh, tp.exactData, 'b-')
plt.plot(tp.mesh, b, 'r.')
plt.tight_layout()
#%%
plt.figure()
results_prior1.plot_ci(95,exact=tp.exactSolution)
plt.title('GMRF prior')
plt.show()

#%%
plt.figure()
results_prior2.plot_ci(95,exact=tp.exactSolution)
plt.title('Cauchy prior')
plt.show()
