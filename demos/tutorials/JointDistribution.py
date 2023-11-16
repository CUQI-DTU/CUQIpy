"""
Joint Distribution tutorial
===========================

    This tutorial shows how to use the Joint Distribution class.

"""
# %%
# Setup
# -----
# We start by importing the necessary modules and loading the model and data.
# The model is a simple 1D convolution, but any model can be used.
# We also define some helper variables to make the code more readable.

import sys; sys.path.append("../..")
import numpy as np
from cuqi.distribution import Gaussian, Gamma, JointDistribution
from cuqi.testproblem import Deconvolution1D

# Model and data
A, y_obs, _ = Deconvolution1D().get_components()

# Model dimensions
n = A.domain_dim
m = A.range_dim

# "Idendity" matricies
In = np.ones(n)
Im = np.ones(m)

# %%
# Bayesian modelling with the Joint Distribution
# ----------------------------------------------
#
# The joint distribution is a tool that allows us to define Bayesian models.
# This is achieved by combining multiple distributions into a single object.
#
# For example, consider the following model
#
# .. math::
#
#   \mathbf{x} &\sim \mathcal{N}(\mathbf{0}, d^{-1} \mathbf{I}_n) \\
#   \mathbf{y} &\sim \mathcal{N}(\mathbf{A} \mathbf{x}, l^{-1} \mathbf{I}_m)
#
# where :math:`\mathbf{A}\in\mathbb{R}^{m\times n}` is a matrix that defines the linear convolution.
# and :math:`d`, :math:`s` are fixed parameters for the prior and noise precision respectively.
#
# We can write this model in CUQIpy as follows:

# Define fixed parameters
d = 100
s = 400

# Define distributions
x = Gaussian(np.zeros(n), 1/d*In)
y = Gaussian(lambda x: A@x, 1/s*Im)

# Define joint distribution p(x,y)
joint = JointDistribution(x, y)

# The joint distributions prints a summary of its components.
print(joint)

# %%
# Basic properties of the joint distribution
# ------------------------------------------
#
# The joint distribution comes with two main properties:
# dim (dimension) and geometry. Because its a joint distribution
# over multiple variables, these are lists.

print(joint.dim)
print(joint.geometry)

# %%
# Density evaluation
# ------------------
#
# A main method of the joint distribution is to be able to evaluate the
# un-normalized log density function (logd) given a set of values for the
# variables. This can be achieved by passing either positional or keyword
# arguments.

# Using keyword arguments
logd = joint.logd(y=np.ones(m), x=np.ones(n))
print(logd)

# Using positional arguments
assert logd == joint.logd(np.ones(m), np.ones(n))

# Here the arguments follow the order shown when printing the joint distribution.
# The order can also be inspected by calling the `.get_parameter_names()` method.
print(joint.get_parameter_names())

# %%
# Conditioning: How to get the posterior distribution
# ---------------------------------------------------
#
# Often we observe some data assumed to come from the Bayesian model
# described by the joint distribution and want to use that data to
# estimate the remaining parameters. To achieve this, we need to define
# the posterior distribution, which is the distribution of the parameters
# given the data. That is, :math:`p(\mathbf{x} \mid \mathbf{y}^{obs})`.
#
# In CUQIpy, this is done by conditioning the joint distribution on the data.
# The conditioning works much more generally than only on the data
# as we will see later.	

# Define posterior distribution
posterior = joint(y=y_obs)

# Notice how the equations and densities change to reflect the conditioning.
print(posterior)

# %%
#
# .. note::
#
#     The posterior distribution as shown above can be sampled via the sampler module.
#     See e.g. the tutorial on sampling :doc:`Sampling tutorial <../_auto_tutorials/4-Samplers>`.
#
# Because the posterior no longer depends on the data, the dimension is only
# with respect to the parameters.
print(f"Posterior dimension: {posterior.dim}")
print(f"Posterior geometry: {posterior.geometry}")

# %%
# Defining hierarchical Bayesian models
# -------------------------------------
#
# The joint distribution is general enough to allow us to define hierarchical
# Bayesian models.
#
# For example, consider the following extension of the Bayesian model earlier:
#
# .. math::
#
#     \begin{align}
#         d &\sim \mathrm{Gamma}(1, 10^{-4}) \\
#         l &\sim \mathrm{Gamma}(1,10^{-4}) \\
#         \mathbf{x} &\sim \mathcal{N}(\mathbf{0}, d^{-1} \mathbf{I}_n) \\
#         \mathbf{y} &\sim \mathcal{N}(\mathbf{A} \mathbf{x}, s^{-1} \mathbf{I}_m)
#     \end{align}
#
# We can write this model in CUQIpy as follows:

# Define distribution
d = Gamma(1, 1e-4)
l = Gamma(1, 1e-4)
x = Gaussian(np.zeros(n), lambda d: 1/d*In)
y = Gaussian(lambda x: A@x, lambda l: 1/l*Im, geometry=m)

# Define joint distribution p(d,l,x,y)
joint_hier = JointDistribution(d, l, x, y)

# Notice how the equations and densities change
print(joint_hier)

# %% Posterior
# We can again define the posterior :math:`p(d, l, \mathbf{x} \mid \mathbf{y}^{obs})`
# as follows:

# Define posterior distribution
posterior_hier = joint_hier(y=y_obs)

# Notice how the density for y becomes a likelihood
print(posterior_hier)

# In this case the posterior is still a joint distribution
# over the parameters d, l, and x.

# %%
#
# .. note::
#
#     The joint distribution as shown above can be sampled via the sampler module.
#     See e.g. the tutorial on Gibbs sampling :doc:`Gibbs tutorial <../_auto_tutorials/Gibbs>`.
#
