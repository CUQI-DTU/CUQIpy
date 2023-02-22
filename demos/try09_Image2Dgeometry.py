# %%
# Load cuqi and other packages
import sys
sys.path.append("..") 
import cuqi
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

#%% Setup exact image with Image2D geometries

N = 128
x_exact_im = cuqi.data.cat(N)
range_geometry = cuqi.geometry.Image2D((N,N), order = "C")
domain_geometry = cuqi.geometry.Image2D((N,N), order = "C")
x_exact = cuqi.array.CUQIarray(x_exact_im, is_par=False, geometry=domain_geometry) # CUQIarray to represent both vector and image
x_exact.plot()
plt.colorbar()

#%% Setup forward problem in CUQI model

# Forward problem. Convolution with kernel k
k = np.array([[1,1,1],[1,1,0],[1,0,0]])
forward_func = lambda x: ndimage.convolve(x, k, mode='constant', cval=0.0)

# Linear cuqi model
model = cuqi.model.Model(forward_func,range_geometry=range_geometry,domain_geometry=domain_geometry)

# %% Compute exact data
b_exact = model.forward(x_exact)
b_exact.plot()
plt.colorbar()

# %% Setup data distribution
data_distb = cuqi.distribution.Gaussian(mean=model, cov=50, geometry=range_geometry) #Geometry is not automatically inferred yet

# %% Sample from data distribution to obtain noisy data
data = data_distb(x = x_exact).sample()
data.plot()
plt.colorbar()
# %%
