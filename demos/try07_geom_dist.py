# %% Initialize and import CUQI
import sys
sys.path.append("..") 
import numpy as np
import cuqi
import matplotlib.pyplot as plt

from cuqi.geometry import Continuous2D

# %% Minimal example

# Import data and forward matrix
A      = np.load("data/Deconvolution.npz")["A"]          #Matrix (numpy)
b      = np.load("data/Deconvolution.npz")["data"]       #Vector (numpy)
m,n    = A.shape

#%%  Set up iid Gaussian size 128, without specifying a geometry, default will be used.
prior      = cuqi.distribution.Gaussian(mean=np.zeros(n), 
                                        sqrtcov=0.1)           #Prior distribution


#%% Generate 100 samples
prior_samples = prior.sample(100)

#%% Plot three of the samples specified by index. This uses plot method of default Continuous1D geometry
prior_samples.plot([1,5,10])

#%%  In a similar way one can also plot chains, here of variable 20 and 80
prior_samples.plot_chain([20,80])

#%% Set up a new Gaussian, this time specifying it to be a 2D image geometry
p2D  = cuqi.distribution.Gaussian(mean=np.zeros(16**2), 
                                  sqrtcov=0.1, 
                                  geometry=Continuous2D(grid=(16,16)))

#%% Generate 100 samples
p2Dsamples = p2D.sample(100)

#%% Note the geometry is included in the samples object
print(p2Dsamples.geometry)

#%%  Plot the 50th sample, note this uses the 2D plot method of Continuous2D
p2Dsamples.plot(50)
plt.colorbar()

#%% Plot 5 randomly selected samples using 2D plot method of Continuous2D as subplots
p2Dsamples.plot() 
plt.colorbar()

#%%  Directly plot the sample mean from samples object, note narrower colorbar
p2Dsamples.plot_mean()
plt.colorbar()
# %%
