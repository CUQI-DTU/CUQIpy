# %% This demo showcases how to call a function in Matlab and use it in CUQIpy.
import sys; sys.path.append('../..')
import cuqi
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
# %%
matlab_instance = cuqi.matlab.MatlabInstance()
# %% Compute the sum of two numpy arrays using matlab
x = np.ones(10)
y = np.ones(10)*2
z = matlab_instance("plus", x, y) # The string is the matlab command
assert np.array_equal(z, x+y)

# %% Let us try a 2d convolution
TP = cuqi.testproblem.Deconvolution2D(phantom=cuqi.data.grains())
X = TP.exactSolution.funvals # 2D Image
PSF = TP.Miscellaneous["PSF"] # 2D PSF

# %% Compute convolution using Matlab
# NB: Requires that Matlab has Image Processing Toolbox

Y_matlab = matlab_instance("conv2", X, PSF, 'same')
plt.imshow(Y_matlab)

# %% Compute convolution using scipy
Y_scipy = sp.signal.convolve2d(X, PSF, mode='same')
plt.imshow(Y_scipy)

assert np.allclose(Y_scipy, Y_matlab) # Same result is produced by both methods


# %% Timing wise the matlab version is actually faster!
%timeit sp.signal.convolve2d(X, PSF, mode='same')
%timeit matlab_instance("conv2", X, PSF, 'same')

# %% Custom functions are also supported
# In this folder we have a function custom .m function

# Compute convolution using custom matlab function 
# Here with boundary conditions (notice lack of artifacts in image)
Y = matlab_instance("custom_conv2", X, PSF, 'symmetric')
plt.imshow(Y)
