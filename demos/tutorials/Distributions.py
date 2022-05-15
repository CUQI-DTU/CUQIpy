"""
1. Distributions
==========================

"""
# %%
import numpy as np
import matplotlib.pyplot as plt
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

#%% Multivariate distributions
pY = cuqi.distribution.Normal(np.linspace(-5,5,100), np.linspace( 1,4,100))

#%% Generate samples
sY = pY.sample(10000)
