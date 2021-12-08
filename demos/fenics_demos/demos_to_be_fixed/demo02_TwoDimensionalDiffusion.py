
#%%
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("../")
import cuqi
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

#%% Set up and solve cuqi problem that uses FEniCSPDEModel 

# Problem of form b = A(x) + e
# Create operator A
A = cuqi.PDEmodel.FEniCSDiffusion2D(measurement_type ='sigma_norm_gradu') 

#%% Create & plot prior 
n = A.dim
pr_mean = np.zeros(n)
prior = cuqi.distribution.GMRF(pr_mean,25,n,1,'zero') 
ps = cuqi.samples.Samples(prior.sample(100))
plt.figure()
ps.plot_ci(95,exact=np.zeros(n))
plt.savefig('fig_prior.png')

#%% Create noise (e) and data (b) 
true_m = prior.sample(1)
true_u = A.forward(true_m)
noise_std = 0.01 * np.max(true_u)
e = cuqi.distribution.Gaussian(np.zeros(n), noise_std, np.eye(n))
b = true_u + e.sample().T 

#%% Create cuqi problem (sets up likelihood and posterior) 
IP = cuqi.problem.Type1(b,A,e,prior)

#%% Sample & plot posterior
results = IP.sample(Ns=10) 
plt.figure()
results.plot_ci(95,exact=true_m)
plt.show()
plt.savefig('fig1.png')

#%% Plot posterior variance
plt.figure()
im = results.plot_sd(A) # when A = None
plt.title('standard deviation')
plt.xlabel('x')
plt.ylabel('y')
axins_p = inset_axes(plt.gca(), width='4%', height='50%', loc=4)
cp_p =plt.colorbar(im, cax = axins_p,  orientation ='vertical')#ticks=p_ticks_list[idx]
plt.show()
plt.savefig('fig2.png')

#%% Plot posterior variance
plt.figure()
im = results.plot_mean(A)
plt.title('mean')
plt.xlabel('x')
plt.ylabel('y')
axins_p = inset_axes(plt.gca(), width='4%', height='50%', loc=4)
cp_p =plt.colorbar(im, cax = axins_p,  orientation ='vertical')#ticks=p_ticks_list[idx]
plt.show()
plt.savefig('fig3.png')
