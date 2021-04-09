import sys
import time

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(".")
import cuqi

# Create X ~ Normal( -1.0,4.0) distirbution
pX = cuqi.distribution.Normal(-1.0, 4.0)

#%% Help is available for getting to know how to use distribution object
help(pX)

#%% And help for specific methods, like sample
help(pX.sample)

#%% Generate a few samples
print(pX.sample())
print(pX.sample())

#%%  Many realizations: Samples and Statistics
sX = pX.sample(100000)
#sX.stats #TODO

#%% Multivariate distributions
pY = cuqi.distribution.Normal(np.linspace(-5,5,100), np.linspace( 1,4,100))

#%% Generate samples
sY = pY.sample(10000)

#----- REST IS TODO -----
# Samples plots with envelope
#figure
#sY.plot_statistics()

# Individual componets
#figure
#sY.plot_selection([1 100])

#%% Define geometry and parameter for GMRF distribution
N = 150       # number of pixels
dom = 1      # 1D or 2D domain
if (dom == 1):
   n = N 
elif (dom == 2):
   n = N**2

# Set up GMRF
mean = np.zeros(n)
prec = 4
pX = cuqi.distribution.GMRF(mean, prec, N, dom, 'neumann')

# evaluation of PDF
Ns = 8
xi = np.random.randn(n, Ns)
Z = pX.pdf(xi)

# call method to sample
sampleX = pX.sample(Ns)

# Display samples - Samples.plot makes different plot depending on dimension
#figure
#sampleX.plot(6)
if (dom == 1):
    tt = np.linspace(0, 1, N)
    plt.figure()
    plt.plot(tt, sampleX)
    plt.xlim(0, 1)
    plt.show()
elif (dom == 2):
    tt = np.linspace(0, 1, N)    
    T, S = np.meshgrid(tt, tt)
    plt.figure()
    for i in range(Ns):
        Z = sampleX[:, i].reshape(N,N)
        c = np.linspace(Z.min(), Z.max(), 6)
        plt.contour(T, S, Z, 7, linewidths=0.25, colors='k')
        plt.contourf(T, S, Z, cmap='Blues')
        cb = plt.colorbar(shrink=0.95, aspect=25, pad = 0.03, ticks=c)
        cb.formatter.set_scientific(True)
        cb.formatter.set_powerlimits((0, 0))
        cb.ax.yaxis.set_offset_position('left')
        cb.update_ticks()
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.pause(1.5)
        plt.clf()
