

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
A = cuqi.PDEmodel.FEniCSDiffusion2D(measurement_type ='potential',parameter_type = "fixed_radius_inclusion") 

#%% Create & plot prior 
n = 2
low = np.array([0,0])
high = np.array([1,1])
prior = cuqi.distribution.Uniform(low,high) 
ps = prior.sample(100)
plt.figure()
ps.plot_ci(95,exact=np.array([.5,.5]))
plt.savefig('fig_prior.png')

#%% Create noise (e) and data (b) 

#true_m = np.array([.4,.3])
true_m = np.array([.42,.32])
true_u = A.forward(true_m)
n_data = len(true_u)
noise_std = 0.01 * np.max(true_u)
e = cuqi.distribution.Gaussian(np.zeros(n_data), noise_std, np.eye(n_data))
b = true_u + e.sample().T 

#%% Create cuqi problem (sets up likelihood and posterior) 
IP = cuqi.problem.Type1(b,A,e,prior)

#%% Sample & plot posterior
results = IP.sample(Ns=100) 
plt.figure()
results.plot_ci(95,exact=true_m)
plt.savefig('fig1.png')

#%% Plot chains
plt.figure()
plt.plot(results.samples[0,:])
plt.plot(results.samples[1,:])
plt.title('chains')
plt.savefig('fig2.png')

#%% Plot True conductivity
plt.figure()
import dolfin as dl
true_m_f = dl.Function(A.Vh[1])
true_m_f.vector().set_local(true_m)
true_kappa = dl.interpolate(A.kappa(true_m_f),A.Vh[0])
X,Y,Z = A.grid4data_plot(true_kappa.vector(), x_res=100, y_res=100)
im = plt.pcolor(X,Y,Z)
plt.title('True kappa')
plt.xlabel('x')
plt.ylabel('y')
axins_p = inset_axes(plt.gca(), width='4%', height='50%', loc=4)
cp_p =plt.colorbar(im, cax = axins_p,  orientation ='vertical')#ticks=p_ticks_list[idx]
plt.savefig('fig3.png')

#%% Plot potential (True, Inferred)
plt.figure()
X,Y,Z = A.grid4data_plot(true_u, x_res=100, y_res=100)
im = plt.pcolor(X,Y,Z)
plt.title('True u')
plt.xlabel('x')
plt.ylabel('y')
axins_p = inset_axes(plt.gca(), width='4%', height='50%', loc=4)
cp_p =plt.colorbar(im, cax = axins_p,  orientation ='vertical')#ticks=p_ticks_list[idx]
plt.savefig('fig4.png')

#%% Plot inferred conductivity
plt.figure()
import dolfin as dl
mean_m_f = dl.Function(A.Vh[1])
mean_m_f.vector().set_local(np.mean(results.samples, axis=1))
true_kappa = dl.interpolate(A.kappa(true_m_f),A.Vh[0])
X,Y,Z = A.grid4data_plot(true_kappa.vector(), x_res=100, y_res=100)
im = plt.pcolor(X,Y,Z)
plt.title('Mean kappa')
plt.xlabel('x')
plt.ylabel('y')
axins_p = inset_axes(plt.gca(), width='4%', height='50%', loc=4)
cp_p =plt.colorbar(im, cax = axins_p,  orientation ='vertical')#ticks=p_ticks_list[idx]
plt.savefig('fig5.png')




# %%
