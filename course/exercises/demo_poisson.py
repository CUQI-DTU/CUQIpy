#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dst, idst
import sys
sys.path.append("../..") 
import cuqi
from poissonmodel import poisson

np.random.seed(seed =0)

#%%
N = 129 # number of spatial KL discretization 
dx = np.pi/N
x = np.linspace(dx/2,np.pi-dx/2,N)
true_alpha = np.exp( 5*x*np.exp(-2*x)*np.sin(np.pi-x) )

#%%
model = poisson(N=N)
y_exact = model.solve_with_conductivity(true_alpha)

#%%
SNR = 100 # signal to noise ratio
sigma = np.linalg.norm(y_exact)/SNR
sigma2 = sigma*sigma # variance of the observation Gaussian noise

#y_obs = y_exact + np.random.normal(0,sigma,y_exact.shape)
#plt.plot(y_obs,label='noisy')
#plt.legend()

y = y_exact

#%%
likelihood = cuqi.distribution.Gaussian(model,sigma,np.eye(N-1))

#%% Prior
prior = cuqi.distribution.Gaussian(np.zeros((N,)),1)

#%%
IP = cuqi.problem.BayesianProblem(likelihood,prior,y)
results = IP.sample_posterior(10000)

results.plot_chain([10, 50, 100])

#%% Plot mean
x_mean = np.mean(results.burnthin(5000,5).samples,axis=-1)

#%%
model.domain_geometry.plot_mapped( x_mean )#, plot_mapped = False)
plt.title("Posterior sampled mean")
#y = np.exp(10*model.domain_geometry.par2fun(x_mean))
#plt.plot(y)
plt.plot(x,true_alpha, label = 'True')
plt.legend()
plt.show()

#%% MAP estimate
def posterior_logpdf(x):
    logpdf = -prior.logpdf(x) - likelihood(x=x).logpdf(y) 
    return logpdf

# Starting point
p0 = np.random.randn(N)

# BFGS MAP
solver = cuqi.solver.minimize(posterior_logpdf, p0)
p_MAP, info_MAP = solver.solve()
alpha_MAP = model.domain_geometry.apply_map(p_MAP)
print('relative error BFGS MAP:', np.linalg.norm(alpha_MAP-true_alpha)/np.linalg.norm(true_alpha))

#%%
model.domain_geometry.plot_mapped( p_MAP )#, plot_mapped = False)
model.domain_geometry.plot_mapped( x_mean )
plt.title("Posterior MAP")
plt.plot(x,true_alpha, label = 'True')
plt.legend()
plt.show()

#%% ML estimate

def likelihood_logpdf(x):
    logpdf = - likelihood(x=x).logpdf(y) 
    return logpdf

# Starting point
p0 = np.random.randn(N)

# BFGS MAP
solver = cuqi.solver.minimize(likelihood_logpdf, p0)
p_ML, info_ML = solver.solve()
alpha_ML = model.domain_geometry.apply_map(p_ML)
print('relative error BFGS MAP:', np.linalg.norm(alpha_ML-true_alpha)/np.linalg.norm(true_alpha))

#%% plot
model.domain_geometry.plot_mapped( p_ML )#, plot_mapped = False)
plt.title("Posterior ML")
plt.plot(x,true_alpha, label = 'True')
plt.legend()
plt.show()
# %%
