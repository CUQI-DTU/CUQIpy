import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../..") 
import cuqi

# domain definition
N = 501            # spatial discretization 
L = 1

# define model 
f = lambda xs: 10*np.exp( -( (xs - 0.5)**2 ) / 0.02) # source term
model = cuqi.model.Poisson_1D(N=N, L=L, source=f, field_type=None)
x = model.domain_geometry

# random field parameters
mean = 0.0
var = 10
lc = 0.2
p = 2
C_YY = lambda x1, x2: var*np.exp( -(1/p) * (abs(x1-x2)/lc)**p )

# forward propagation
Ns = 50

# 1. random field: point-wise 
XX, YY = np.meshgrid(x, x, indexing='ij')
Sigma = C_YY(XX, YY)
prior = cuqi.distribution.GaussianCov(mean*np.ones(N), Sigma)
#
Y = prior.sample(Ns).samples
kappa = np.exp(Y) # conductivity field realizations
u = np.empty((N-1, Ns))  # store pressure field realizations
for i in range(Ns):
    u[:, i] = model.forward(kappa[:, i])

mu_Y = np.mean(Y, axis=1, dtype=np.float64)
sigma_Y = np.std(Y, ddof=0, axis=1, dtype=np.float64)
mu_kappa = np.mean(kappa, axis=1, dtype=np.float64)
sigma_kappa = np.std(kappa, ddof=0, axis=1, dtype=np.float64)
mu_u = np.mean(u, axis=1, dtype=np.float64)
sigma_u = np.std(u, ddof=0, axis=1, dtype=np.float64)

plt.figure(1)
ax1 = plt.subplot(121)
ax1.plot(x, kappa)
ax1.plot(x, mu_kappa, 'k-', linewidth=2)
ax1 = plt.subplot(122)
ax1.plot(x[:-1], u)
plt.tight_layout()
plt.show()

# 2. random field: KL expansion 
d_KL = N
geom = cuqi.geometry.CustomKL(x, C_YY, mean, np.sqrt(var), trunc_term=d_KL)
#
theta = np.random.normal(0, 1, size=(d_KL, Ns)) # KL coefficients
kappa = np.empty((N, Ns))  # store conductivity field realizations
u = np.empty((N-1, Ns))  # store pressure field realizations
for i in range(Ns):
    kappa[:, i] = np.exp(geom.par2fun(theta[:, i]))
    u[:, i] = model.forward(kappa[:, i])
    
mu_kappa = np.mean(kappa, axis=1)
sigma_kappa = np.std(kappa, ddof=0, axis=1)
mu_u = np.mean(u, axis=1)
sigma_u = np.std(u, ddof=0, axis=1)

plt.figure(2)
ax1 = plt.subplot(121)
ax1.plot(x, kappa)
ax1.plot(x, mu_kappa, 'k-', linewidth=2)
ax1 = plt.subplot(122)
ax1.plot(xs, u)
plt.tight_layout()
plt.show()