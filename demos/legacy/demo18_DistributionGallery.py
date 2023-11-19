#%%

import sys
sys.path.append("../")
import time
import numpy as np
import matplotlib.pyplot as plt

# myfuns
from cuqi.sampler import pCN, MH, NUTS
from cuqi.distribution import DistributionGallery

#%%
# ==========================================================================
# choose benchmark
# ==========================================================================
exa = 4
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

# evaluate gradient
if dist.gradient_func is not None:
    Xg, Yg = np.meshgrid(np.linspace(xmin, xmax, ng), np.linspace(ymin, ymax, ng))
    Xfg, Yfg = Xg.flatten(), Yg.flatten()
    posg = np.vstack([Xfg, Yfg]).T  
    grad = dist.gradient(posg)
    norm = np.linalg.norm(grad, axis=0)
    u, v = grad[0, :]/norm, grad[1, :]/norm

    # plot PDF and gradient
    plt.figure(3)
    plt.contourf(X, Y, Z, nl)
    plt.contour(X, Y, Z, nl, linewidths=0.5, colors='k') 
    plt.quiver(posg[:, 0], posg[:, 1], u, v, units='xy', scale=4, color='gray')
# plt.pause(1)

#%% 
# =============================================================================
# sample benchmarks
# =============================================================================
# run MH and NUTS samplers
Ns = int(5e3)      # number of samples
Nb = int(0.5*Ns)   # burn-in

# MH
MCMC = MH(dist)
ti = time.time()
x_s_MH = MCMC.sample_adapt(Ns, Nb)
print('Elapsed time MH:', time.time() - ti, '\n')

# plots MH
plt.figure(1)
plt.contourf(X, Y, Z, nl)
plt.contour(X, Y, Z, nl, linewidths=0.5, colors='k') 
plt.plot(x_s_MH.samples[0, :], x_s_MH.samples[1, :], 'r.', markersize=1.5, alpha=0.3)
plt.gca().set_aspect('equal', adjustable='box')
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
plt.title('MH samples')
#
plt.figure(2)
x_s_MH.plot_chain([0, 1])
plt.xlim(0, Ns)
plt.title('MH chains')

# NUTS
if dist.gradient_func is not None:
    MCMC = NUTS(dist, opt_acc_rate=0.8)    
    ti = time.time()
    x_s_NUTS = MCMC.sample_adapt(Ns, Nb)
    print('Elapsed time NUTS:', time.time() - ti, '\n')

    # plot NUTS
    plt.figure(3)
    plt.plot(x_s_NUTS.samples[0, :], x_s_NUTS.samples[1, :], 'r.', markersize=1.5, alpha=0.3)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.title('NUTS samples')
    #
    plt.figure(4)
    x_s_NUTS.plot_chain([0, 1])
    plt.xlim(0, Ns)
    plt.title('NUTS chains')
    plt.show()
