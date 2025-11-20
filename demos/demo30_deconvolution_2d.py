#%% This file show-cases the Deconvolution 2D test problem.
import sys
sys.path.append("..")
import numpy as np
import matplotlib.pyplot as plt
import cuqi

#For defining the precision matricies in Gaussian at the bottom of file.
from scipy.sparse import diags, kron, eye 

# %% 2D Deconvolution testproblem
# It can simply be defined as any other testproblem
TP = cuqi.testproblem.Deconvolution2D() #Default values

# %% Of particular interest is the option to add numpy arrays for the phantom (sharp image) to try blurring other images
# We provide a suite of examples in cuqi.data. They all have a `size` argument. For example:
phantom = cuqi.data.grains(size=128)        # Image with big piece-wise constant regions
phantom = cuqi.data.shepp_logan(size=128)   # A standard test image modelling a simple human head
phantom = cuqi.data.camera(size=128)        # Real image of man with camera
phantom = cuqi.data.cat(size=128)           # Photo of a cat
phantom = cuqi.data.satellite(size=128)     # Photo of a satellite
# There exist many more. Have e.g. a look at https://cuqi.gitlab.io/cuqipy/cuqi.data.html

# The sharp image is added when defining the testproblem by passing the `phantom` argument:
# We pass dim=128 here also. If the phantom image is different than dim the testproblem will automatically resize.
# This is mostly relevant for custom images as all images in cuqi.data have a size argument.
# If you have a color image use `cuqi.data.rbg2gray` to convert it to a gray-scale image.
TP = cuqi.testproblem.Deconvolution2D(dim=128, phantom=phantom)

# %% We can plot the sharp image using the `plot` method that our CUQIarray objects have.
TP.exactSolution.plot() #Sharp image

# %% Similar for the blurred image
TP.data.plot()

# %% We can also extract properties like the PSF (this is just a 2D numpy array)
plt.imshow(TP.Miscellaneous["PSF"], cmap="gray")

# %% Deblur image in Bayesian framework
# To deblur the image using a Bayesian approach we need a likelihood and prior.
# The likelihood is defined automatically for us in the testproblem (TP.likelihood)
# so we only need to define the prior.

# Gaussian i.i.d. prior (identity matrix)
TP.prior = cuqi.distribution.Gaussian(
    mean=np.zeros(TP.model.domain_dim),
    cov=1
)

# LMRF prior (promotes piece-wise constant regions)
TP.prior = cuqi.distribution.LMRF(
    location=0,
    scale=0.01,
    geometry=TP.model.domain_geometry # Prior geometry should match domain geometry of model
)

# For Gaussians with correlation structure (see further below)

# %% After defining the prior we sample the posterior.
# Here the sampler is automatically chosen for us.
samples = TP.sample_posterior(200)

# %% After computing the samples we can plot statistics like the mean
samples.plot_mean()

# %% or standard deviation
samples.plot_std()

# %% Defining Gaussians with correlation structure.

# Helper variables
n = TP.model.domain_dim # Number of parameters
N = TP.model.domain_geometry.fun_shape[0] # Number of parameters in one dimension

# Scaling factor of precision matrix
alpha = 100 

# Here we define a precision matrix with a banded diagonal structure.
P = alpha*diags([-1, 2, -1], [-1, 0, 1], shape=(N, N)) # Maximum flatness
#P = alpha*diags([1, -4, 6, -4, 1], [-2, -1, 0, 1, 2], shape=(N, N)) # Minimum roughness

# We now extend the precision matrix to match the 2D problem using kronecker products.
I = eye(N, dtype=int)
Ds = kron(I, P)
Dt = kron(P, I)
P = Ds+Dt # Precision matrix defined for the 2D problem

# Then we define the Gaussian using Gaussian precision to avoid computing the (dense) covariance matrix.
TP.prior = cuqi.distribution.Gaussian(mean=np.zeros(n), prec=P, geometry=TP.model.domain_geometry)

# We can plot some samples of this prior
TP.prior.sample(5).plot()

# %% Finally we can sample the posterior using this prior
samples = TP.sample_posterior(200)

# %% And compute e.g. the mean
samples.plot_mean()

# %% More options for structures Gaussians can be found by using the GMRF-type priors
TP.prior = cuqi.distribution.GMRF(
    mean=np.zeros(TP.model.domain_dim),
    prec=100,
    bc_type="zero", # Try e.g. zero, neumann or periodic.
    order=2, # Try e.g. order 1 or 2 (higher order is more smooth)
    geometry=TP.model.domain_geometry
)

# We can plot some samples of this prior
TP.prior.sample(5).plot()

# %% Finally we can sample the posterior using this prior
samples = TP.sample_posterior(200)

# %% And compute e.g. the mean
samples.plot_mean()

# %% and standard deviation
samples.plot_std()

#%% Test problem with options explained

TP = cuqi.testproblem.Deconvolution2D(
    dim=128, # The 2D image size (dim x dim)
    PSF = "Gauss", # The point spread function type. Can also be custom numpy array.
    PSF_param=2.56, # The parameter of the PSF. Larger parameter => more blur.
    PSF_size=21, # The size of the PSF. Everything outside is assumed to be zero.
    BC = "periodic", # Boundary conditions of blurring operation.
    phantom = "satellite", # The phantom (sharp image). Can also be custom numpy 2d array.
    noise_type = "Gaussian", # The type of noise (Only Gaussian supported for now).
    noise_std = 0.0036, # The standard deviation of the noise.
)
