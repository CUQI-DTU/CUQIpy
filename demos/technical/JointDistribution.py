"""
Joint Distribution techinical demo.
===================================

    This shows technical aspects of the JointDistribution class.
    These are mostly relevant for developers and advanced users.
    See :doc:`../auto_tutorials/JointDistribution` for a more
    user-friendly introduction.

"""
# %%
# Setup
# -----

import sys; sys.path.append("../..")
import numpy as np
from cuqi.distribution import GaussianCov, Gamma, JointDistribution
from cuqi.testproblem import Deconvolution1D

# Model and data
A, y_obs, _ = Deconvolution1D.get_components()

# Model dimensions
n = A.domain_dim
m = A.range_dim

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
x = GaussianCov(np.zeros(n), lambda d: d)
y = GaussianCov(lambda x: A@x, lambda l: l)

# Define joint distribution p(d,l,x,y)
joint = JointDistribution([d, l, x, y])
print(joint)

# %% Define posterior
posterior = joint(y=y_obs)

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

# %%
# Finally you can evaluate the log density of the stacked using a single vector
# of parameters.

logd = posterior_stacked.logd(np.ones(posterior_stacked.dim))

print(logd)
