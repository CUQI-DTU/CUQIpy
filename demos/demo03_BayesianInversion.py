# %%
import sys
import time
sys.path.append("../")

import numpy as np
import matplotlib.pyplot as plt

import cuqi

#%% %Cuqi deblur test problem
tp = cuqi.testproblem.Deblur() #Default values

#%% Plot true, blurred and noisy data
plt.figure
plt.plot(tp.t,tp.f_true)
plt.plot(tp.t,tp.g_true)
plt.plot(tp.t,tp.data)
plt.ylim([-0.5,3.5])
plt.legend(['True','Blurred','Blurred and noisy'])

#%% Unpack problem in b = A*x + e
b = tp.data;    #%Measured data
A = tp.model    # Class with model
e = tp.noise    # Class with noise

#%% A few additional parameters from test problem needed
h = tp.meshsize;        # Size of mesh elements
n = tp.model.dim[1];    # Number of unknowns

#%% Two choices of prior

prior1 = cuqi.distribution.GMRF(np.zeros(n),25,n,1,'zero')

loc = np.zeros(n)
delta = 1
scale = delta*h
prior2 = cuqi.distribution.Cauchy_diff(loc, scale, 'neumann')

#%% Generate and display some prior samples
sp1 = prior1.sample(5)
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

#Problem structure b = A(x)+e represented as Problem.type1 so far
IP = cuqi.problem.Type1(b,A,e,prior1)

#cuqi.Problem simply sets up likelihood and posterior for us
results_prior1 = IP.sample(Ns) 


#%% 2.  "Absolute non-expert level" -  just ask for UQ!

#The blur TestProblem is a subclass of cuqi.Problem, just need to add prior
tp.prior = prior2

#Use UQ convenience method:
results_prior2 = tp.sample(Ns)

#%%
norm_f = np.linalg.norm(tp.f_true)
med_xpos1 = np.median(results_prior1.samples, axis=1) # sp.stats.mode
sigma_xpos1 = results_prior1.samples.std(axis=1)
relerr = round(np.linalg.norm(med_xpos1 - tp.f_true)/norm_f*100, 2)
print('\nGMRF Relerror median:', relerr, '\n')

med_xpos2 = np.median(results_prior2.samples, axis=1) # sp.stats.mode
sigma_xpos2 = results_prior2.samples.std(axis=1)
relerr = round(np.linalg.norm(med_xpos2 - tp.f_true)/norm_f*100, 2)
print('\nCauchy Relerror median:', relerr, '\n')

#%%
plt.figure()
plt.plot(tp.t, tp.f_true, 'k-')
plt.plot(tp.t, tp.g_true, 'b-')
plt.plot(tp.t, b, 'r.')
plt.tight_layout()
#%%
plt.figure()
results_prior1.plot_ci(95,exact=tp.f_true)
plt.title('GMRF prior')
plt.show()

#%%
plt.figure()
results_prior2.plot_ci(95,exact=tp.f_true)
plt.title('Cauchy prior')
plt.show()
