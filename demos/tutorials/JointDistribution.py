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

import sys; sys.path.append("..")
import numpy as np
from cuqi.distribution import GaussianCov, Gamma, JointDistribution
from cuqi.testproblem import Deconvolution1D

# Model and data
A, y_obs, _ = Deconvolution1D.get_components()

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
#   \mathbf{x} &\sim \mathcal{N}(\mathbf{0}, d^2 \mathbf{I}_n) \\
#   \mathbf{y} &\sim \mathcal{N}(\mathbf{A} \mathbf{x}, s^2 \mathbf{I}_m)
#
# where :math:`\mathbf{A}\in\mathbb{R}^{m\times n}` is a matrix that defines the linear convolution.
# and :math:`d`, :math:`s` are fixed parameters for the prior and noise standard deviations respectively.
#
# We can write this model in CUQIpy as follows:

# Define fixed parameters
d = 0.1
s = 0.05

# Define distributions
x = GaussianCov(np.zeros(n), d*In)
y = GaussianCov(lambda x: A@x, s*Im)

# Define joint distribution p(x,y)
joint = JointDistribution([x, y])

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
#         \mathbf{x} &\sim \mathcal{N}(\mathbf{0}, d^2 \mathbf{I}_n) \\
#         \mathbf{y} &\sim \mathcal{N}(\mathbf{A} \mathbf{x}, s^2 \mathbf{I}_m)
#     \end{align}
#
# We can write this model in CUQIpy as follows:

# Define distribution
d = Gamma(1, 1e-4)
l = Gamma(1, 1e-4)
x = GaussianCov(np.zeros(n), lambda d: d*In)
y = GaussianCov(lambda x: A@x, lambda l: l*Im)

# Define joint distribution p(d,l,x,y)
joint = JointDistribution([d, l, x, y])

# Notice how the equations and densities change
print(joint)

# %% Posterior
# We can again define the posterior :math:`p(d, l, \mathbf{x} \mid \mathbf{y}^{obs})`
# as follows:

# Define posterior distribution
posterior = joint(y=y_obs)

# Notice how the density for y becomes a likelihood
print(posterior)

# %%
# Enabling Gibbs sampling
# -----------------------
#
# One of the main purposes of the joint distribution is to be able to sample
# using a Gibbs scheme. For this to work we need to be able to define the
# distribution of each variable conditioned on the other variables.
#
# That is, we need to define
#
# .. math::
#
#       C_\mathbf{x}&=p(\mathbf{x} \mid \hat{\mathbf{y}}, \hat{d}, \hat{l})\\
#       C_d&=p(d \mid \hat{\mathbf{y}}, \hat{\mathbf{x}}, \hat{l})\\
#       C_l&=p(l \mid \hat{\mathbf{y}}, \hat{\mathbf{x}}, \hat{d})
#
# Assuming we have some values for :math:`\mathbf{x}`, :math:`d` and :math:`l`,
# we can simply condition the joint distribution on these values and it will
# handle the rest.

# Assume we want to condition on these values
yh = y_obs
xh = np.ones(n)
dh = 1
lh = 1

# The conditionals can be computed as follows:
Cx = joint(y=yh, d=dh, l=lh)
Cd = joint(y=yh, x=xh, l=lh)
Cl = joint(y=yh, x=xh, d=dh)

# We can try inspecting one of these conditional distributions.
# Notice how the equations and densities change to reflect the conditioning.
# In particular how the equation reflects a likelihood+prior structure.
print(Cd)

# %%
# Going into the internals of the joint distribution
# --------------------------------------------------
#
# One of the major tricks we currently employ to make the joint distribution
# work with our samplers is to allow reducing the joint distribution to a
# single Density (Distribution, Likelihood or Posterior). This is because
# our samplers are implemented to work with these.
# 
# The way this is achieved is by calling a private method as shown below,
# when conditioning. The samplers can then directly use the resulting Density.

Cd._reduce_to_single_density()

# %%
# Another "hack" is to have the joint distribution act as if it were a single
# parameter density. This is achieved by calling the `._as_stacked()` method.
# This returns a new "stacked" joint distribution that the samplers/solvers
# can use as if it were any other Density.

posterior_stacked = posterior._as_stacked()

print(posterior_stacked)

# %% 
# Here the dimension is the sum of the dimensions of the individual variables.
# The geometry is assumed as a default geometry matching the dimension.
print(f"Stacked Posterior dimension: {posterior_stacked.dim}")
print(f"Stacked Posterior geometry: {posterior_stacked.geometry}")
