import sys
import time
sys.path.append("../")

import numpy as np
import matplotlib.pyplot as plt

import cuqi

#%% Create X ~ Normal( -1.0,4.0) distirbution
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
pY = cuqi.distribution.Normal(np.linspace(-5,5,100),np.linspace( 1,4,100))

#%% Generate samples
sY = pY.sample(10000)

#%%
#----- REST IS TODO -----
#%% Samples plots with envelope
#figure
#sY.plot_statistics()

#%% Individual componets
#figure
#sY.plot_selection([1 100])

#%% Define geometry and parameter for GMRF distribution
#N   = 64;     % number of pixels
#dom = 1;      % 1D or 2D domain

#if dom == 1
#    n = N;   
#elseif dom == 2
#    n = N^2;
#end

#mean = zeros(n,1);
#prec = 4;

#%% Set up GMRF Probability Distribution
#pX   = cuqi.ProbDistb.GMRF(mean, prec, N, dom, 'zero')  % zero, periodic, neumann

#%% call method to sample 
#sampleX = pX.sample(10)

#%% Display samples - Samples.plot makes different plot depending on dimension
#figure
#sampleX.plot(6)