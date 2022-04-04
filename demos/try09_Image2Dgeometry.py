# %%
# Load cuqi and other packages
import sys
sys.path.append("..") 
import cuqi
import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
import scipy.sparse as sps

#%%
# Forward matrix
A = np.zeros((5,10))
for i in range(0,5):
    A[i,i] = 1

# Use Image geometries
range_geometry = cuqi.geometry.Image2D((5,10))
domain_geometry = cuqi.geometry.Image2D((10,10))

# Linear cuqi model
model = cuqi.model.LinearModel(A,range_geometry=range_geometry,domain_geometry=domain_geometry)

#%% Setup exact parameters x
x_exact = cuqi.samples.CUQIarray(np.ones((10,10)), is_par=True, geometry=domain_geometry)
x_exact.plot()

# %% Compute exact data
b_exact = model.forward(x_exact)
b_exact.plot()

# %% Setup data distribution
data_distb = cuqi.distribution.GaussianCov(mean = model, cov = 0.1)

# %% Sample from data distribution to obtain noisy data
data = data_distb(x = x_exact).sample() # breaks
# %%
