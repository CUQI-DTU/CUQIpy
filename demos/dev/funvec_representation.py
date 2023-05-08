"""
funvec variable representation supported by CUQIpy Geometry classes
====================================================================

    In this demo we illustrate the utility of an introduced representation of variables in CUQIpy (besides the already introduced parameter and the function value representation). This representation, called funvec, is a vector represnetation of the function values. The function values generally can be an n-dimensional array or any other type of object. When possible, funvec is used to represent the function values as a vector (1D) array. funvec, is meant to be used internally by other CUQIpy classes and is not expected to be of interest to the typical user. In particular, funvec is used to represent the function values of samples in a vector format. This is useful for computing statistics on the function values of samples. 
"""
# %%
# Import the necessary modules
# -----------------------------

import cuqi
import numpy as np
import matplotlib.pyplot as plt

# %% 
# Create a 1D and a mapped 2D geometries
# --------------------------------------

#%%
# Create a continuous 1D geometry
geom_1D = cuqi.geometry.Continuous1D(10)

#%%
# Create a mapped 2D Image geometry
geom_2D = cuqi.geometry.MappedGeometry(
    cuqi.geometry.Image2D((3, 3)),
    map=lambda x: x**2)

#%%
# Query the shape of the  function value for both geometries
print(geom_1D.fun_shape)
print(geom_2D.fun_shape)

# %%
# Create random samples for each geometry
# -----------------------------------------

samples_1D = cuqi.samples.Samples(np.random.rand(10, 5), geometry=geom_1D)
samples_2D = cuqi.samples.Samples(np.random.rand(9, 5), geometry=geom_2D)

# %%
# Query the geometries `has_funvec` property
print(geom_1D.has_funvec)
print(geom_2D.has_funvec)

#%%
# Note that only the 2D geometry has a funvec representation. The 1D
# geometry does not need this representation because the function values
# are already represented as a vector (1D) array.

# %%
# Create funvals samples 
# ----------------------

samples_1D_funvals = samples_1D.funvals
samples_2D_funvals = samples_2D.funvals

# %%
# query the shape of the funvals for both geometries
print(samples_1D_funvals.shape)
print(samples_2D_funvals.shape)

# %%
# Note that the funvals shape for the 2D geometry is (9, 5) and not (3, 3, 5).
# To understand why this is the case, we query the funvals samples 
# _funvals_directly_supported property.
print(samples_1D_funvals._funvals_directly_supported)
print(samples_2D_funvals._funvals_directly_supported)

# %%
# Note that the 1D samples funvals are directly supported because par2fun
# results in a 1D array of function values. However, funvals for 2D samples is
# not directly supported because par2fun results in a 2D array representing the 
# function values.
#
# However, Samples class can use fun2funvec methods of the geometry to convert to 
# function value to a vector representation of the these values. In the Image2D
# geometry, this is done by flattening the 2D array of function values, thus
# we see that the shape of the funvals is (9, 5) and not (3, 3, 5).
# 
# The flag is_funvec is used to indicate whether the samples funvals are in 
# funvec representation or not (i.e. obtained using fun2funvec methods or not). 

print(samples_1D_funvals._is_funvec)
print(samples_2D_funvals._is_funvec)

# %%
# Computing statistics on parameter and on function values for samples (the 1D geometry case)
# --------------------------------------------------------------------------------------------

# %%
# Compute mean on parameter values and on function values
plt.figure()
samples_1D.plot_mean(color='r')
samples_1D_funvals.plot_mean(color='b', linestyle='--')

# %%
# Compute variance on parameter values and on function values
plt.figure()
samples_1D.plot_variance(color='r')
samples_1D_funvals.plot_variance(color='b', linestyle='--')

# %% 
# Note that in both cases the mean and the variance are the same. This is because computing the mean and variance on the parameter values then converting the results to function values is equivalent to computing the mean and variance on the function values in this case.

# %%
# Computing statistics on parameter and on function values (funvec) for samples (the 2D geometry case)
# ----------------------------------------------------------------------------------------------------

# %%
# Compute mean on parameter and function values
plt.figure()
samples_2D.plot_mean()
plt.colorbar()
plt.figure()
samples_2D_funvals.plot_mean()
plt.colorbar()

# %%
# Compute variance on parameter and function values
plt.figure()
samples_2D.plot_variance()
plt.colorbar()
plt.figure()
samples_2D_funvals.plot_variance()
plt.colorbar()

# %%
# Note that in both cases the mean and variance are not the same. This is because computing the mean and variance on the parameter values then converting the results to a function value is not equivalent to computing the mean and variance on the function values in this case. Also, the mean and variance of the function values are computed on the vector representation of the function values.
