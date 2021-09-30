# =============================================================================
# Created by:
# Felipe Uribe @ DTU
# =============================================================================
# Version 2020-10
# =============================================================================
#%%
import sys
sys.path.append("../")
import time
import numpy as np
import matplotlib.pyplot as plt

# myfuns
from cuqi.sampler import pCN, RWMH
from cuqi.distribution import Gaussian, UserDefinedDistribution, Gallery
from cuqi.samples import Samples
#%%
# ==========================================================================
# sample a Gaussian
# ==========================================================================
 
# target function to sample
dist = Gallery("CalSom91")
#dist = Gallery("BivariateGaussian")

def target(x): return dist.logpdf(x)
#%%
m, n = 200, 200
X, Y = np.meshgrid(np.linspace(-3, 3, m), np.linspace(-3, 3, n))
Xf, Yf = X.flatten(), Y.flatten()
pos = np.vstack([Xf, Yf]).T   # pos is (m*n, d)
Z = dist.pdf(pos).reshape((m, n))

plt.figure()
plt.contourf(X, Y, Z, 10)
plt.contour(X, Y, Z, 4, colors='k')
plt.gca().set_aspect('equal', adjustable='box')

#%%
# =============================================================================
# reference measure (or proposal)
# =============================================================================
d = 2
mu = np.zeros(d)
ref = Gaussian(mu, np.ones(d), np.eye(d))   # standard Gaussian

# =============================================================================
# posterior sampling
# =============================================================================
scale = 0.1
x0 = 0.5*np.ones(d)
MCMC_pCN = pCN(ref, target, scale, x0)
MCMC_RWMH = RWMH(ref, target, scale, x0)

# run sampler
Ns = int(1e4)      # number of samples
Nb = int(0.2*Ns)   # burn-in
#
ti = time.time()
x_s_RWMH, target_eval, acc = MCMC_RWMH.sample_adapt(Ns, Nb)
print('Elapsed time:', time.time() - ti)
#%%
plt.figure()
plt.contourf(X, Y, Z, 4)
plt.contour(X, Y, Z, 4, colors='k')
plt.gca().set_aspect('equal', adjustable='box')
plt.plot(x_s_RWMH.samples[0,:], x_s_RWMH.samples[1,:], 'b.', alpha=0.3)
#

plt.figure()
x_s_RWMH.plot_chain(0)

#%%
ti = time.time()
x_s_pCN, target_eval2, acc2 = MCMC_pCN.sample_adapt(Ns, Nb)
print('Elapsed time:', time.time() - ti)

plt.figure()
plt.contourf(X, Y, Z, 4)
plt.contour(X, Y, Z, 4, colors='k')
plt.gca().set_aspect('equal', adjustable='box')
plt.plot(x_s_pCN.samples[0,:], x_s_pCN.samples[1,:], 'b.', alpha=0.3)
#

plt.figure()
x_s_pCN.plot_chain(0)