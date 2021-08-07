# %% Initialize and import CUQI
import sys
sys.path.append("..") 
import numpy as np
import cuqi
import matplotlib.pyplot as plt

# %%
x = cuqi.distribution.Normal("x",5,1)
x.sample()

#%%
y = cuqi.distribution.Normal("y",0,None)
y.sample() #Gives error if value is unspecified

# %%
y(std=10).sample()

# %%
z = cuqi.distribution.Normal("z",0,lambda sigma: np.sqrt(sigma))
z(sigma=2).sample()
# %%
# Example from Johns book. Algorithm 5.1
n_samp = 1000
alpha = 1
beta = 1e-4

# Load deconvolution test problem
tp = cuqi.testproblem.Deconvolution()
m,n = tp.model.dim

# Matricies
A = tp.model.get_matrix()
b = tp.data
L = np.eye(n)

#x = cuqi.distribution.Gaussian(mean=None,std=None)
l = cuqi.distribution.Gamma(shape=m/2+alpha,rate=None)
d = cuqi.distribution.Gamma(shape=n/2+alpha,rate=None)

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
    ls[k+1] = l(rate=.5*np.linalg.norm(A@xs[:,k]-b)**2+beta).sample()
    ds[k+1] = d(rate=.5*xs[:,k].T@(L@xs[:,k])+beta).sample()

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
