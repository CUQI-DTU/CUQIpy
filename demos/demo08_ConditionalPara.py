# %% Initialize and import CUQI
import sys
sys.path.append("..") 
import numpy as np
import cuqi
import matplotlib.pyplot as plt

# %%
# A simple normal distribution with mean 5 and std 1
y = cuqi.distribution.Normal(mean=5,std=1)
y.sample()

# %%
# We can change the value if internal parameters of the distribution as follows
y2 = y(std=10)
y2.sample()

# %%
# We can omit one of the inputs to create an "incomplete" distribution,
# that is one way to represent a conditional distribution
y3 = cuqi.distribution.Normal(mean=5)

# %%
# A conditional distribution can NOT be sampled. We must first specify
# a value for the conditioning variable.

#  y3.sample()   # This gives an error
y3(std=10).sample()

# %% 
# Specifying the missing parameter creates a new "complete" distribution
# which could also be saved in a separate variable
z3 = y3(std=10)
z3.sample()

# %%
# We can specify parameters as callable functions/methods
# and condition directly on the parameters of those methods
z = cuqi.distribution.Normal(0,lambda sigma: np.sqrt(sigma))
z(sigma=2).sample()

# %%
#Functions for mean and std with various (shared) inputs
mean = lambda sigma,gamma: sigma+gamma
std  = lambda delta,gamma: np.sqrt(delta+gamma)

z = cuqi.distribution.Normal(mean,std, geometry=1)
Z = z(sigma=3,gamma=-2)
Z = Z(delta=5)
Z.sample()

#%%
# We can also specify an optional name of distribution (to be used by certain sampling algorithms which require distributions to interconnect)
y = cuqi.distribution.Normal(mean=5, std=1,name="y")
y.name

# %%
# Example from Johns book. Algorithm 5.1
n_samp = 5000
alpha = 1
beta = 1e-4

# Load deconvolution test problem
tp = cuqi.testproblem.Deconvolution1D()
m = tp.model.range_dim
n = tp.model.domain_dim

# Matricies
A = tp.model.get_matrix()
b = tp.data
L = np.eye(n)

# Define hyperpriors
l = cuqi.distribution.Gamma(shape=m/2+alpha,rate=lambda x: .5*np.linalg.norm(A@x-b)**2+beta)
d = cuqi.distribution.Gamma(shape=n/2+alpha,rate=lambda x: .5*x.T@(L@x)+beta)

# Define prior
mean_func = lambda l,d: np.linalg.solve(l*A.T@A+d*L,l*A.T@b)
cov_func  = lambda l,d: np.linalg.inv(l*A.T@A+d*L)
x = cuqi.distribution.Gaussian(mean=mean_func,cov=cov_func, geometry=n)

# Preallocate sample vectors
ls = np.zeros(n_samp+1)
ds = np.zeros(n_samp+1)
xs = np.zeros((n,n_samp+1))

# Initial parameters
ls[0] = 1/0.05; ds[0]=1/0.02

# Initial x
xs[:,0] = x(l=ls[0],d=ds[0]).mean

for k in range(n_samp):

    #Sample hyperparameters conditioned on x
    ls[k+1] = l(x=xs[:,k]).sample()
    ds[k+1] = d(x=xs[:,k]).sample()

    # Sample x conditioned on l,d
    xs[:,k+1] = x(l=ls[k+1],d=ds[k+1]).sample()


# %%
plt.subplot(121); plt.plot(1/ls); plt.title("lambda chain")
plt.subplot(122); plt.plot(1/ds); plt.title("delta chain")
# %%
burn_in = 1000
thin = 1
xsamples = cuqi.sampler.Samples(xs[:,burn_in:thin:-1])
xsamples.plot_ci(95,exact=tp.exactSolution)

# %%
