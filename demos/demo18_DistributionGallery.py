#%%

import sys
sys.path.append("../")
import time
import numpy as np
import matplotlib.pyplot as plt

# myfuns
from cuqi.sampler import pCN, MetropolisHastings, NUTS
from cuqi.distribution import DistributionGallery

#%%
# ==========================================================================
# choose benchmark
# ==========================================================================
exa = 2
if exa == 0:
    xmin, xmax = -4, 4
    ymin, ymax = -4, 4    
    dist = DistributionGallery("BivariateGaussian")
elif exa == 1:
    xmin, xmax = -7, 7
    ymin, ymax = -9, 6
    dist = DistributionGallery("funnel")
elif exa == 2:
    xmin, xmax = -4, 4
    ymin, ymax = -4, 4
    dist = DistributionGallery("mixture")
elif exa == 3:
    xmin, xmax = -4, 4
    ymin, ymax = -4, 4
    dist = DistributionGallery("squiggle")
elif exa == 4:
    xmin, xmax = -4, 4
    ymin, ymax = -4, 4
    dist = DistributionGallery("donut") # similar to CalSom91
elif exa == 5:
    xmin, xmax = -6, 6
    ymin, ymax = -4, 4
    dist = DistributionGallery("banana")
elif exa == 6:
    xmin, xmax = -2, 2
    ymin, ymax = -2, 2    
    dist = DistributionGallery("CalSom91")

#%% 
# ==========================================================================
# Plots densities
# ==========================================================================
m, n, nl, ng = 300, 300, 30, 30   # discretization

# evaluate PDF
X, Y = np.meshgrid(np.linspace(xmin, xmax, m), np.linspace(ymin, ymax, n))
Xf, Yf = X.flatten(), Y.flatten()
pts = np.vstack([Xf, Yf]).T   # pts is (m*n, d)
Z = dist.pdf(pts).reshape((m, n))

# plot PDF
plt.figure(1)
plt.contourf(X, Y, Z, nl)
plt.contour(X, Y, Z, nl, linewidths=0.5, colors='k') 

# evaluate gradient
if dist.gradient_func is not None:
    Xg, Yg = np.meshgrid(np.linspace(xmin, xmax, ng), np.linspace(ymin, ymax, ng))
    Xfg, Yfg = Xg.flatten(), Yg.flatten()
    posg = np.vstack([Xfg, Yfg]).T  
    grad = dist.gradient(posg)
    norm = np.linalg.norm(grad, axis=0)
    u, v = grad[0, :]/norm, grad[1, :]/norm
    plt.quiver(posg[:, 0], posg[:, 1], u, v, units='xy', scale=5, color='gray')
# plt.pause(1)

#%% 
# =============================================================================
# sample benchmarks
# =============================================================================
# use MetropolisHastings if gradients are not available, and NUTS if they are
if dist.gradient_func is not None:
    MCMC = NUTS(dist)
else:
    MCMC = MetropolisHastings(dist)
    
# run sampler
Ns = int(5e3)      # number of samples
Nb = int(0.5*Ns)   # burn-in
#
ti = time.time()
x_s = MCMC.sample_adapt(Ns, Nb)
print('Elapsed time:', time.time() - ti)

#%%
# ==========================================================================
# Plots samples and chains
# ==========================================================================
plt.figure(1)
plt.plot(x_s.samples[0, :], x_s.samples[1, :], 'r.', markersize=1, alpha=0.3)
plt.gca().set_aspect('equal', adjustable='box')
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
#
plt.figure(2)
x_s.plot_chain([0, 1])
plt.xlim(0, Ns)
plt.show()