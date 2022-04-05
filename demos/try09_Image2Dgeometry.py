# %%
# Load cuqi and other packages
import sys
sys.path.append("..") 
import cuqi
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

#%%
# Forward problem. Convolution with kernel k
k = np.array([[1,1,1],[1,1,0],[1,0,0]])
forward_func = lambda x: ndimage.convolve(x, k, mode='constant', cval=0.0)
adjoint_func = lambda x: x

# Use Image geometries
range_geometry = cuqi.geometry.Image2D((4,4), order = "C")
domain_geometry = cuqi.geometry.Image2D((4,4), order = "C")

# Linear cuqi model
model = cuqi.model.LinearModel(forward_func,adjoint=adjoint_func,range_geometry=range_geometry,domain_geometry=domain_geometry)

# %% Setup exact parameters x
x_exact_im = np.array([[1, 2, 0, 0],
              [5, 3, 0, 4],
              [0, 0, 0, 7],
              [9, 3, 0, 0]])
x_exact = cuqi.samples.CUQIarray(x_exact_im, is_par=False, geometry=domain_geometry) # CUQIarray to represent both vector and image
x_exact.plot()
plt.colorbar()

# %% Compute exact data
b_exact = model.forward(x_exact)
b_exact.plot()
plt.colorbar()

# %% Setup data distribution
data_distb = cuqi.distribution.GaussianCov(mean = model, cov = 1, geometry=range_geometry) #Geometry is not automatically inferred yet

# %% Sample from data distribution to obtain noisy data
data = data_distb(x = x_exact).sample()
data.plot()
plt.colorbar()
# %%
