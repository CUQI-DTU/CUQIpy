# %%
import sys
import time
sys.path.append("..")

import numpy as np
import matplotlib.pyplot as plt

import cuqi

#%% %Cuqi deblur test problem
tp = cuqi.testproblem.Deblur() #Default values

#%% Plot true, blurred and noisy data
plt.figure
plt.plot(tp.t,tp.exactSolution)
plt.plot(tp.t,tp.exactData)
plt.plot(tp.t,tp.data)
plt.ylim([-0.5,3.5])
plt.legend(['True','Blurred','Blurred and noisy'])

#%% Unpack problem
b = tp.data;        # Measured data
A = tp.model        # Class with model
L = tp.likelihood   # Class with likelihood

#%% A few additional parameters from test problem needed
h = tp.meshsize;        # Size of mesh elements
n = tp.model.dim[1];    # Number of unknowns

#%% Two choices of prior

P1 = cuqi.distribution.GMRF(np.zeros(n),25,n,1,'zero')

loc = np.zeros(n)
delta = 1
scale = delta*h
P2 = cuqi.distribution.Cauchy_diff(loc, scale, 'neumann')

#%% Generate and display some prior samples
sp1 = P1.sample(5)
#sp2 = prior2.sample(5)

plt.figure
plt.subplot(1,2,1)
plt.plot(sp1)
plt.title('GMRF')
#TODO
#subplot(1,2,2)
#sp2.plot(5)
#title('Cauchy')

#%% Number of samples
Ns = 2000

#%% 1.  "High level"  - set up cuqi Problem

# Define Bayesian model
IP = cuqi.problem.BayesianModel(likelihood=L,prior=P1,model=A,data=b)

# Then we can simply sample the posterior
results = IP.sample_posterior(Ns) 

# Plot 95% confidence interval
results.plot_ci(95,exact=tp.exactSolution)

#%% 2.  "Absolute non-expert level" -  just ask for UQ!

#The blur TestProblem is a subclass of cuqi.Problem, just need to add prior
tp.prior = P2

#Use UQ convenience method:
tp.UQ()