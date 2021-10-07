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
from cuqi.sampler import pCN, MetropolisHastings
from cuqi.distribution import Gaussian, Posterior, DistributionGallery
from cuqi.samples import Samples
#%%
# ==========================================================================
# sample a Gaussian
# ==========================================================================
 
dist = DistributionGallery("BivariateGaussian")
#dist = DistributionGallery("CalSom91")


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
#d = 2
#mu = np.zeros(d)
#ref = Gaussian(mu, np.ones(d), np.eye(d))   # standard Gaussian

# =============================================================================
# posterior sampling
# =============================================================================


MCMC_MH = MetropolisHastings(dist)


# run sampler
Ns = int(1e4)      # number of samples
Nb = int(0.2*Ns)   # burn-in
#
ti = time.time()
x_s_MH, target_eval, acc = MCMC_MH.sample_adapt(Ns, Nb)
print('Elapsed time:', time.time() - ti)
#%%
plt.figure()
plt.contourf(X, Y, Z, 4)
plt.contour(X, Y, Z, 4, colors='k')
plt.gca().set_aspect('equal', adjustable='box')
plt.plot(x_s_MH.samples[0,:], x_s_MH.samples[1,:], 'b.', alpha=0.3)
#

plt.figure()
x_s_MH.plot_chain(0)

#%%
