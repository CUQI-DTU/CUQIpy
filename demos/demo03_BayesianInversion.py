import sys
import time
sys.path.append("../")

import numpy as np
import matplotlib.pyplot as plt

import cuqi

#%% %Cuqi deblur test problem
tp = cuqi.TestProblem.Deblur() #Default values

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

#prior1 = cuqi.Distribution.GMRF(np.zeros(n,1), 25, n, 1, 'zero')

loc = np.zeros(n)
delta = 1
scale = delta*h
prior2 = cuqi.Distribution.Cauchy_diff(loc, scale, 'neumann')

#%% Generate and display some prior samples
#sp1 = prior1.sample(5)
#sp2 = prior2.sample(5)

#figure
#subplot(1,2,1)
#sp1.plot(5)
#title('GMRF')
#subplot(1,2,2)
#sp2.plot(5)
#title('Cauchy')

#%% Number of samples
Ns = 1000

#%% 1.  "High level"  - set up cuqi Problem

#Problem structure b = A(x)+e represented as Problem.type1 so far
IP = cuqi.Problem.Type1(b,A,e,prior2)
#%%
#cuqi.Problem simply sets up likelihood and posterior for us
results = IP.sample(Ns) 


#%% 2.  "Absolute non-expert level" -  just ask for UQ!

#The blur TestProblem is a subclass of cuqi.Problem, just need to add prior
tp.prior = prior2

#Use UQ convenience method:
UQresults = tp.sample(Ns)

#%%
norm_f = np.linalg.norm(tp.f_true)
med_xpos = np.median(results, axis=1) # sp.stats.mode
sigma_xpos = results.std(axis=1)
lo95, up95 = np.percentile(results, [2.5, 97.5], axis=1)
relerr = round(np.linalg.norm(med_xpos - tp.f_true)/norm_f*100, 2)
print('\nRelerror median:', relerr, '\n')

#%%
plt.figure()
plt.plot(tp.t, tp.f_true, 'k-')
plt.plot(tp.t, tp.g_true, 'b-')
plt.plot(tp.t, b, 'r.')
plt.tight_layout()

plt.figure()
plt.plot(tp.t, tp.f_true, '-', color='forestgreen', linewidth=3, label='True')
plt.plot(tp.t, med_xpos, '--', color='crimson', label='median')
plt.fill_between(tp.t, up95, lo95, color='dodgerblue', alpha=0.25)
plt.legend(loc='upper right', shadow=False, ncol = 1, fancybox=True, prop={'size':15})
plt.xticks(np.linspace(tp.t[0], tp.t[-1], 5))
plt.xlim([tp.t[0], tp.t[-1]])
plt.ylim(-0.5, 3.5)
plt.tight_layout()
# plt.savefig('fig.png', format='png', dpi=150, bbox_inches='tight')
plt.show()