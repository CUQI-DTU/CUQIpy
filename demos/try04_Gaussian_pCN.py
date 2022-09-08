# =============================================================================
# Created by:
# Felipe Uribe @ DTU
# =============================================================================
# Version 2020-10
# =============================================================================
import sys
sys.path.append("../")
import time
import numpy as np
import matplotlib.pyplot as plt

# myfuns
from cuqi.sampler import pCN, RWMH
from cuqi.distribution import Gaussian

# ==========================================================================
# sample a Gaussian
# ==========================================================================
d = 200
mu = np.zeros(d)
sigma = np.linspace(0.5, 1, d)

# target function to sample
dist = Gaussian(mu, sigma**2)
def target(x): return dist.logpdf(x)

# =============================================================================
# reference measure (or 'prior' if it is a BIP)
# =============================================================================
ref = Gaussian(mu, np.ones(d))   # standard Gaussian

# =============================================================================
# posterior sampling
# =============================================================================
scale = 0.1
x0 = 0.5*np.ones(d)
MCMC = pCN(ref, target, scale, x0)
MCMC2 = RWMH(ref, target, scale, x0)

# run sampler
Ns = int(1e4)      # number of samples
Nb = int(0.2*Ns)   # burn-in
#
ti = time.time()
x_s = MCMC.sample_adapt(Ns, Nb)
target_eval = x_s.loglike_eval
acc = x_s.acc_rate
print('Elapsed time:', time.time() - ti)
#
ti = time.time()
x_s2 = MCMC2.sample_adapt(Ns, Nb)
target_eval2 = x_s2.loglike_eval
acc2 = x_s2.acc_rate
print('Elapsed time:', time.time() - ti)

# =============================================================================
mu_xpos = x_s.mean(axis=1) 
sigma_xpos = x_s.std(axis=1)
lo95, up95 = np.percentile(x_s, [2.5, 97.5], axis=1)

mu_xpos2 = x_s2.mean(axis=1) 
sigma_xpos2 = x_s2.std(axis=1)
lo952, up952 = np.percentile(x_s2, [2.5, 97.5], axis=1)

# =============================================================================
# plots
# =============================================================================
# pCN
fig, (ax3, ax1, ax2) = plt.subplots(1, 3, figsize=(16, 5))
ax3.plot(target_eval)
ax3.set_xlabel('Sample index')
ax3.set_ylabel('Target function values (pCN)')
ax1.plot(sigma, mu_xpos, 'b.')
ax1.set_ylabel('sample mean at each coordinate')
ax1.set_xlim(0.5,1)
ax1.set_ylim(-0.6,0.6)
ax2.plot(sigma, sigma_xpos, 'b.')
ax2.set_ylabel('sample std at each coordinate')
ax2.set_xlim(0.5,1)
ax2.set_ylim(0,1.1)

# RW
fig, (ax3, ax1, ax2) = plt.subplots(1, 3, figsize=(16, 5))
ax3.plot(target_eval2)
ax3.set_xlabel('Sample index')
ax3.set_ylabel('Target function values (RW)')
ax1.plot(sigma, mu_xpos2, 'r.')
ax1.set_ylabel('sample mean at each coordinate')
ax1.set_xlim(0.5,1)
ax1.set_ylim(-0.6,0.6)
ax2.plot(sigma, sigma_xpos2, 'r.')
ax2.set_ylabel('sample std at each coordinate')
ax2.set_xlim(0.5,1)
ax2.set_ylim(0,1.1)
plt.tight_layout()

plt.show()
