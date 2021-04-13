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
from cuqi.diagnostics import Geweke

# ==========================================================================
# sample a Gaussian
# ==========================================================================
d = 10
mu = np.zeros(d)
sigma = np.linspace(0.5, 1, d)
R = np.eye(d)

# target function to sample
dist = Gaussian(mu, sigma, R)
def target(x): return dist.logpdf(x)

# =============================================================================
# reference measure (or 'prior' if it is a BIP)
# =============================================================================
ref = Gaussian(mu, np.ones(d), R)   # standard Gaussian

# =============================================================================
# posterior sampling
# =============================================================================
scale = 0.1
x0 = 0.25*np.ones(d)
MCMC = RWMH(ref, target, scale, x0)

# run sampler
Ns = int(2e4)      # number of samples
Nb = int(0.2*Ns)   # burn-in
#
ti = time.time()
x_s, target_eval, acc = MCMC.sample_adapt(Ns, Nb)
print('Elapsed time:', time.time() - ti)

# =============================================================================
# stat
mu_xpos = x_s.mean(axis=1) 
sigma_xpos = x_s.std(axis=1)
lo95, up95 = np.percentile(x_s, [2.5, 97.5], axis=1)

# Geweke
_, p = Geweke(x_s.T) # we want p>=0.95

# =============================================================================
# plots
# =============================================================================
# RW
fig, (ax3, ax1, ax2) = plt.subplots(1, 3, figsize=(16, 5))
ax3.plot(target_eval)
ax3.set_xlabel('Sample index')
ax3.set_ylabel('Target function values (RWM)')
ax1.plot(sigma, mu_xpos, 'b.')
ax1.set_ylabel('sample mean at each coordinate')
ax1.set_xlim(0.5,1)
ax1.set_ylim(-0.6,0.6)
ax2.plot(sigma, sigma_xpos, 'b.')
ax2.set_ylabel('sample std at each coordinate')
ax2.set_xlim(0.5,1)
ax2.set_ylim(0,1.1)

plt.show()