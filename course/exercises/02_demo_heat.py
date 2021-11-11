# %%
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../..") 
import cuqi

# domain definition
N = 50            # spatial discretization 
L = 1
T = 0.2

# random field parameters
mean = 0.0
var = 10
lc = 0.2
p = 2
C_YY = lambda x1, x2: var*np.exp( -(1/p) * (abs( x1 - x2 )/lc)**p )
KL_map = lambda x: x
# %%
# define model 
model = cuqi.testproblem.Heat_1D(dim=N, endpoint=L, max_time=T, field_type=None, KL_map=KL_map).model
x = model.domain_geometry.grid

# forward propagation
Ns = 100

# 1. random field: point-wise 
XX, YY = np.meshgrid(x, x, indexing='ij')
Sigma = C_YY(XX, YY)
prior = cuqi.distribution.GaussianCov(mean*np.ones(N), Sigma)
#
Y = prior.sample(Ns).samples
u = np.empty((N, Ns))  # store pressure field realizations
for i in range(Ns):
    u[:, i] = model.forward(Y[:, i])
kappa = model.domain_geometry.par2fun(Y)
#
mu_kappa = np.mean(kappa, axis=1)
sigma_kappa = np.std(kappa, ddof=0, axis=1)
mu_u = np.mean(u, axis=1)
sigma_u = np.std(u, ddof=0, axis=1)

plt.figure(1)
ax1 = plt.subplot(121)
ax1.plot(x, kappa)
ax1.plot(x, mu_kappa, 'k-', linewidth=2)
ax1 = plt.subplot(122)
ax1.plot(x, u)
plt.tight_layout()
# %%
# 2. random field: KL expansion sine basis 
model = cuqi.testproblem.Heat_1D(dim=N, endpoint=L, max_time=T, field_type='KL', KL_map=KL_map).model
x = model.domain_geometry.grid

d_KL = 50

theta = np.random.normal(0, 1, size=(d_KL, Ns)) # KL coefficients
u = np.empty((N, Ns))  # store pressure field realizations
kappa = np.empty((N, Ns))
for i in range(Ns):
    kappa[:, i] = model.domain_geometry.par2fun(theta[:, i])
    u[:, i] = model.forward(theta[:, i])
    
mu_kappa = np.mean(kappa, axis=1)
sigma_kappa = np.std(kappa, ddof=0, axis=1)
mu_u = np.mean(u, axis=1)
sigma_u = np.std(u, ddof=0, axis=1)

plt.figure(2)
ax1 = plt.subplot(121)
ax1.plot(x, kappa)
ax1.plot(x, mu_kappa, 'k-', linewidth=2)
ax1 = plt.subplot(122)
ax1.plot(x, u)
plt.tight_layout()
# %%
# 3. random field: KL expansion custom


field = cuqi.geometry.CustomKL(grid=x,cov_func=C_YY, mean=mean, std=np.sqrt(var), trunc_term=d_KL,mapping=KL_map)
model = cuqi.testproblem.Heat_1D(dim=N, endpoint=L, max_time=T, field_type=field).model
x = model.domain_geometry.grid
#
theta = np.random.normal(0, 1, size=(d_KL, Ns)) # KL coefficients
u = np.empty((N, Ns))  # store pressure field realizations
kappa = np.empty((N, Ns))
for i in range(Ns):
    kappa[:, i] = model.domain_geometry.apply_map(theta[:, i])
    u[:, i] = model.forward(theta[:, i])
    
mu_kappa = np.mean(kappa, axis=1)
sigma_kappa = np.std(kappa, ddof=0, axis=1)
mu_u = np.mean(u, axis=1)
sigma_u = np.std(u, ddof=0, axis=1)

plt.figure(3)
ax1 = plt.subplot(121)
ax1.plot(x, kappa)
ax1.plot(x, mu_kappa, 'k-', linewidth=2)
ax1 = plt.subplot(122)
ax1.plot(x, u)
plt.tight_layout()

#%%
# 4. random field: step function
model = cuqi.testproblem.Heat_1D(dim=N, endpoint=L, max_time=T, field_type='Step', KL_map=KL_map).model
x = model.domain_geometry.grid
#
theta = np.random.normal(0, 1, size=(3, Ns)) # KL coefficients
u = np.empty((N, Ns))  # store pressure field realizations
kappa = np.empty((N, Ns))
for i in range(Ns):
    kappa[:, i] = model.domain_geometry.par2fun(theta[:, i])
    u[:, i] = model.forward(theta[:, i])
    
mu_kappa = np.mean(kappa, axis=1)
sigma_kappa = np.std(kappa, ddof=0, axis=1)
mu_u = np.mean(u, axis=1)
sigma_u = np.std(u, ddof=0, axis=1)

plt.figure(4)
ax1 = plt.subplot(121)
ax1.plot(x, kappa)
ax1.plot(x, mu_kappa, 'k-', linewidth=2)
ax1 = plt.subplot(122)
ax1.plot(x, u)
plt.tight_layout()
plt.show()