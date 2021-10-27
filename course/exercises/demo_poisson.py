#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dst, idst
import sys
sys.path.append("../..") 
import cuqi
from poissonmodel import poisson

#%%
N = 129 # number of spatial KL discretization 
dx = np.pi/N
x = np.linspace(dx/2,np.pi-dx/2,N)
true_alpha = np.exp( 5*x*np.exp(-2*x)*np.sin(np.pi-x) )

#%%
model = poisson(N=N)
y_obs = model.solve_with_conductivity(true_alpha)

#%%
SNR = 100 # signal to noise ratio
sigma = np.linalg.norm(y_obs)/SNR
sigma2 = sigma*sigma # variance of the observation Gaussian noise

#%%
likelihood = cuqi.distribution.Gaussian(model,sigma,np.eye(N-1))

#%% Prior
prior = cuqi.distribution.Gaussian(np.zeros((N,)),1)

#%%
IP = cuqi.problem.BayesianProblem(likelihood,prior,y_obs)
results = IP.sample_posterior(5000)

#%% Plot mean
x_mean = np.mean(results.samples,axis=-1)

#plt.plot(x_mean)
#plt.title('Posterior mean of parameters')

#%%
model.domain_geometry.plot( x_mean, case="mapped" )
plt.title("Posterior mean")

plt.plot(x,true_alpha)
plt.show()
# %%
