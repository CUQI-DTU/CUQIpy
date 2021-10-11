

#%%
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("../")
import cuqi
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

#%% Set up and solve cuqi problem that uses FEniCSPDEModel 
A = cuqi.PDEmodel.FEniCSDiffusion2D(measurement_type ='potential',parameter_type = "fixed_radius_inclusion") 

#%% Create & plot prior 
low = np.array([0,0])
high = np.array([1,1])
prior = cuqi.distribution.Uniform(low,high) 
ps = prior.sample(100)
ps.geometry = cuqi.geometry.Discrete(variables=['x','y'])
plt.figure()
ps.plot_ci(95,exact=np.array([.5,.5]))

#%% Create noise and data 
true_m = np.array([.42,.32])
true_u = A.forward(true_m)
noise_std = 0.01 * np.max(true_u)
likelihood = cuqi.distribution.Gaussian(A, noise_std, np.eye(len(true_u)))
b = likelihood(x=true_m).sample()

#%% Create cuqi Bayesian problem 
IP = cuqi.problem.BayesianProblem(likelihood,prior,b)

#%% Sample & plot posterior
results = IP.sample_posterior(Ns=1000) 
results.geometry = ps.geometry
plt.figure()
results.plot_ci(95,exact=true_m)

#%% Plot chains
plt.figure()
results.plot_chain([0,1])
plt.title('chains')

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

#%% Plot potential (True, Inferred)
plt.figure()
X,Y,Z = A.grid4data_plot(true_u, x_res=100, y_res=100)
im = plt.pcolor(X,Y,Z)
plt.title('True u')
plt.xlabel('x')
plt.ylabel('y')
axins_p = inset_axes(plt.gca(), width='4%', height='50%', loc=4)
cp_p =plt.colorbar(im, cax = axins_p,  orientation ='vertical')#ticks=p_ticks_list[idx]

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