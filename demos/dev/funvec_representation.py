"""
Support vector representation of function values in CUQIpy geometries and samples
====================================================================================

In this demo, we illustrate the utility of an introduced representation of variables in CUQIpy (besides the already introduced parameter and the function value representation). This representation, is a vector representation of the function values. The function values generally can be an n-dimensional array or any other type of object. When needed and when possible, this introduced representation is used to represent the function values as a vector (1D) array. It is meant to be used internally by other CUQIpy classes and is not expected to be of interest to the typical user. In particular, it is used to represent the function values of samples in a vector format. This is useful for computing statistics on the function values of samples. 
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

# %%
# Query the geometries `fun_is_array` property
print(geom_1D.fun_is_array)
print(geom_2D.fun_is_array)

#%%
# This property indicates whether the function value representation is an 
# array (not necessarily a 1D array) or not. 

#%%
# Query the `shape` of the function value representation for both geometries
print(geom_1D.fun_shape)
print(geom_2D.fun_shape)

# %%
# Create random samples corresponding to each geometry
# ----------------------------------------------------

#%%
# Create samples of random parameter values for each geometry
samples_1D = cuqi.samples.Samples(np.random.rand(10, 5), geometry=geom_1D)
samples_2D = cuqi.samples.Samples(np.random.rand(9, 5), geometry=geom_2D)

# %%
# Obtain the corresponding samples of function values (`funvals`)  
# ---------------------------------------------------------------

samples_1D.funvals
samples_2D.funvals

# %%
# Note that the `funvals` property of the `Samples` class returns a `Samples`
# object

# %%
# query the `shape` of the `.funvals` property for both samples objects

print("samples_1D.funvals.shape")
print(samples_1D.funvals.shape)
print("samples_2D.funvals.shape")
print(samples_2D.funvals.shape)

# %% 
# Note that for the 2D geometry, the `shape` of the `funvals` is (3, 3, 5). This
# is because the function value representation is a 2D array.

# %%
# query the `shape` of the `.funvals.vector` property for both samples objects

print("samples_1D.funvals.vector.shape")
print(samples_1D.funvals.vector.shape)
print("samples_2D.funvals.vector.shape")
print(samples_2D.funvals.vector.shape)

# %%
# Now for both geometries, the samples in the converted Samples object is a two
# dimensional arrays. In this case, executing the `funvals` property leads to 
# calling the `par2fun` method of the geometry to convert the parameter values
# to function values. Then executing the `vector` property leads to calling the
# `fun2vec` method of the geometry (if implemented) to convert the function 
# values to a vector representation of these values.

# %%
# query the `shape` of the `.vector` property for both geometries

print("samples_1D.vector.shape")
print(samples_1D.vector.shape)
print("samples_2D.vector.shape")
print(samples_2D.vector.shape)

# %% 
# For samples_1D, we query the `is_par` and `is_vec` properties for all 
# representations of the samples

print("samples_1D.is_par")
print(samples_1D.is_par)
print("samples_1D.is_vec")
print(samples_1D.is_vec)

print("samples_1D.vector.is_par")
print(samples_1D.vector.is_par)
print("samples_1D.vector.is_vec")
print(samples_1D.vector.is_vec)

print("samples_1D.funvals.is_par")
print(samples_1D.funvals.is_par)
print("samples_1D.funvals.is_vec")
print(samples_1D.funvals.is_vec)

print("samples_1D.funvals.vector.is_par")
print(samples_1D.funvals.vector.is_par)
print("samples_1D.funvals.vector.is_vec")
print(samples_1D.funvals.vector.is_vec)

# %%
# Note that for this 1D geometry, the `is_vec` property is always `True`. This
# is because the function value representation is a 1D array.

# %% 
# We do the same for samples_2D

print("samples_2D.is_par")
print(samples_2D.is_par)
print("samples_2D.is_vec")
print(samples_2D.is_vec)

print("samples_2D.vector.is_par")
print(samples_2D.vector.is_par)
print("samples_2D.vector.is_vec")
print(samples_2D.vector.is_vec)

print("samples_2D.funvals.is_par")
print(samples_2D.funvals.is_par)
print("samples_2D.funvals.is_vec")
print(samples_2D.funvals.is_vec)

print("samples_2D.funvals.vector.is_par")
print(samples_2D.funvals.vector.is_par)
print("samples_2D.funvals.vector.is_vec")
print(samples_2D.funvals.vector.is_vec)

# %%
# Note that the property `samples_2D.funvals.is_vec` is false, but after 
# converting the function values to a vector representation using the `vector` 
# property, the property `samples_2D.vector.funvals.vector.is_vec` is true. 

# %%
# Computing statistics on parameter and on function values for samples (the 1D
# geometry case)
# --------------------------------------------------------------------------------------------

# %%
# Compute and plot mean on parameter values and on function values
print("samples_1D.plot_mean")
plt.figure()
samples_1D.plot_mean(color='r')
print("samples_1D.funvals.plot_mean")
plt.figure()
samples_1D.funvals.plot_mean(color='b', linestyle='--')

# %%
# Compute variance on parameter values and on function values
print("samples_1D.plot_variance")
plt.figure()
samples_1D.plot_variance(color='r')
print("samples_1D.funvals.plot_variance")
plt.figure()
samples_1D.funvals.plot_variance(color='b', linestyle='--')

# %% 
# Note that in both cases the mean and the variance computed on parameter values
# are the same as the ones computed on function values. This is because the map
# from parameter values to function values is identity in this case. 

# %%
# Computing statistics on parameter and on function values for samples
# (the 2D geometry case)
# ----------------------------------------------------------------------------------------------------

# %%
# Compute mean on parameter values, function values, and vector representation 
# of function values
print("samples_2D.plot_mean")
plt.figure()
samples_2D.plot_mean()
plt.colorbar()

print("samples_2D.funvals.plot_mean")
plt.figure()
samples_2D.funvals.plot_mean()
plt.colorbar()

print("samples_2D.funvals.vector.plot_mean")
plt.figure()
samples_2D.funvals.vector.plot_mean()
plt.colorbar()

# %%
# Compute variance on parameter values, function values, and vector representation 
# of function values
print("samples_2D.plot_variance")
plt.figure()
samples_2D.plot_variance()
plt.colorbar()

print("samples_2D.funvals.plot_variance")
plt.figure()
samples_2D.funvals.plot_variance()
plt.colorbar()

print("samples_2D.funvals.vector.plot_variance")
plt.figure()
samples_2D.funvals.vector.plot_variance()
plt.colorbar()

# %%
# Note that in both cases the mean and variance computed on parameter values are
# not the same as the ones computed on function values. This is because 
# computing the mean and variance on the parameter values then converting the 
# results to a function value is not equivalent to computing the mean and 
# variance directly on the function values in this case, due to the nonlinear 
# mapping `lambda x: x**2`. Also note that, in this case, we are able to compute
# the statistics directly on the function values, but in some geometries, this
# may not be possible and we may need to use the vector representation of the
# function values to compute the statistics.

# Plotting samples
# ------------------------------------
# Plot samples from `samples_2D`:
print("samples_2D.plot")
plt.figure()
samples_2D.plot([1, 2, 3, 4])

# %%
# Plot samples from `samples_2D.funvals`:
print("samples_2D.funvals.plot")
plt.figure()
samples_2D.funvals.plot([1, 2, 3, 4])

# %%
# Plot samples from `samples_2D.funvals.vector`:
print("samples_2D.funvals.vector.plot")
plt.figure()
samples_2D.funvals.vector.plot([1, 2, 3, 4])

# %%
# The results are the same in all the three cases. Note that for the first case,
# the default behavior of the `plot` method is to convert the samples to the
# function values and then plot them. If one is interested in plotting the
# original parameter values, one can pass `plot_par=True` to the `plot` method.
print("samples_2D.plot([1, 2, 3, 4], plot_par=True)")
plt.figure()
samples_2D.plot([1, 2, 3, 4], plot_par=True)
