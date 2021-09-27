# %% Initialize and import CUQI
import sys
sys.path.append("..") 
import numpy as np
import cuqi
import matplotlib.pyplot as plt

# %%
# A simple normal distribution with mean 5 and std 1
x = cuqi.distribution.Normal(5,1)
x.sample()

#%%
# We can specify an optional name of distribution (to be used by certain sampling algorithms which require distributions to interconnect)
y = cuqi.distribution.Normal(5,1,name="y")
y.name

# %%
# We can change the value if internal parameters of the distribution as follows
y2 = y(std=10)
y2.sample()

# %%
# We can specify parameters as callable functions/methods and condition directly on the parameters of those methods
z = cuqi.distribution.Normal(0,lambda sigma: np.sqrt(sigma))
z(sigma=2).sample()

# %%
#Functions for mean and std with various (shared) inputs
mean = lambda sigma,gamma: sigma+gamma
std  = lambda delta: np.sqrt(delta)

z = cuqi.distribution.Normal(mean,std)
z(sigma=2,delta=5,gamma=-5).sample()
# %%
# Example from Johns book. Algorithm 5.1
n_samp = 1000
alpha = 1
beta = 1e-4

# Load deconvolution test problem
tp = cuqi.testproblem.Deconvolution()
m = tp.model.range_dim
n = tp.model.domain_dim

# Matricies
A = tp.model.get_matrix()
b = tp.data
L = np.eye(n)

#x = cuqi.distribution.Gaussian(mean=None,std=None)
l = cuqi.distribution.Gamma(shape=m/2+alpha,rate=lambda x: .5*np.linalg.norm(A@x-b)**2+beta)
d = cuqi.distribution.Gamma(shape=n/2+alpha,rate=lambda x: .5*x.T@(L@x)+beta)

# Preallocate sample vectors
ls = np.zeros(n_samp+1)
ds = np.zeros(n_samp+1)
xs = np.zeros((n,n_samp+1))

# Initial parameters
ls[0] = 0.05; ds[0]=0.2

# Initial x
mean_x = lambda l,d: np.linalg.inv(l*A.T@A+d*L)@(l*A.T@b)
cov_x  = lambda l,d: np.linalg.inv(l*A.T@A+d*L)
xs[:,0] = mean_x(ls[0],ds[0])

for k in range(n_samp):

    #Sample hyperparameters
    ls[k+1] = l(x=xs[:,k]).sample()
    ds[k+1] = d(x=xs[:,k]).sample()

    # Sample x
    S = np.linalg.cholesky(cov_x(ls[k+1],ds[k+1]))
    xs[:,k+1] = mean_x(ls[k+1],ds[k+1]) + S@np.random.randn(n)
    
# %%
plt.subplot(121); plt.plot(ls); plt.title("lambda chain")
plt.subplot(122); plt.plot(ds); plt.title("delta chain")
# %%
burn_in = 200
plt.plot(tp.exactSolution)
plt.plot(np.mean(xs[:,burn_in:],axis=1))

# %%
