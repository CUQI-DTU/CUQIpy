"""
2D Deconvolution
================

In this example we show how to quantify the uncertainty of a solution to a 2D deconvolution problem.
"""
# %% 
# First we import the modules needed.

import numpy as np
from cuqi.testproblem import Deconvolution2D
from cuqi.distribution import Gaussian, Laplace_diff
from cuqi.problem import BayesianProblem
# %%
# Step 1: Deterministic model
# ---------------------------
#
# Consider the deterministic inverse problem
#
# .. math::
#
#   \mathbf{y} = \mathbf{A} \mathbf{x}
#
# where :math:`\mathbf{A}` is a matrix representing a 2D convolution operation and
# :math:`\mathbf{y}` and :math:`\mathbf{x}` are the data and unknown (solution to the inverse problem) respectively.
#
# A linear forward model like :math:`\mathbf{A}` is represented by a :class:`~cuqi.model.LinearModel`
# and any data (like some observed data :math:`\mathbf{y}^\mathrm{obs}`) as a :class:`~cuqi.samples.CUQIarray`.
#
# The easiest way to get these two components is to use the built-in testproblems.
# Let us extract the model and data for a 2D deconvolution.

A, y_obs, info = Deconvolution2D.get_components()

# %%
# Step 2: Likelihood model
# ------------------------
#
# Suppose our data is corrupted by a Gaussian noise so our observational model is
#
# .. math::
#
#   \mathbf{y}\mid \mathbf{x} \sim \mathcal{N}(\mathbf{A} \mathbf{x}, \sigma^2),
#
# where :math:`\sigma^2` is a noise variance that we know.
#
# We can represent :math:`\mathbf{y}\mid \mathbf{x}` as a :class:`cuqi.distribution.Distribution` object.
# We call the distribution of :math:`\mathbf{y}\mid \mathbf{x}` the data distribution.

data_dist = Gaussian(mean=A, cov=0.01)

# %%
# In actuality we are interested in conditioning the distribution on the observed data :math:`\mathbf{y}^\mathrm{obs}`.
# Mathematically, this produces a likelihood function defined as
#
# .. math::
#
#   \mathcal{L}(\mathbf{x}|\mathbf{y}^\mathrm{obs}) := p(\mathbf{y}^\mathrm{obs}|\mathbf{x}).
#
# This is easily done using the :meth:`cuqi.distribution.Distribution.to_likelihood` method.

likelihood = data_dist.to_likelihood(y_obs)

# %%
# Step 3: Prior model
# -------------------
#
# Now we aim to represent our prior knowledge of the unknown image. In this case, let us assume
# that the unknown is piecewise constant. This can be modelled by assuming a Laplace difference
# prior. The Laplace difference prior can be defined as
#
# .. math::
#
#   \mathbf{x}_{i,j}-\mathbf{x}_{i',j} \sim \mathrm{Laplace}(0, \delta),\\
#   \mathbf{x}_{i,j}-\mathbf{x}_{i,j'} \sim \mathrm{Laplace}(0, \delta),
#
# where :math:`\delta` is the scale parameter defining how likely jumps from one pixel value
# to another are in the horizontal and vertical directions.
# 
# This distribution comes pre-defined in CUQIpy as the :class:`cuqi.distribution.Laplace_diff`.
# Notice we have to specify the physical dimensions of the unknown.

prior = Laplace_diff(location=np.zeros(A.domain_dim), scale=0.1, physical_dim=2)

# %%
# Step 4: Posterior sampling
# --------------------------
#
# Given the likelihood and prior we can construct a :class:`cuqi.problem.BayesianProblem` and
# sample the posterior. Notice that a well-suited sampler is automatically chosen based on the
# model, likelihood and prior chosen.

samples = BayesianProblem(likelihood, prior).sample_posterior(200)

# %%
# Step 5: Posterior analysis
# --------------------------
#
# Finally, after sampling we can analyze the posterior. There are many options here. For example,
# we can plot the credible intervals for the unknown image and compare it to the true image.

ax = samples.plot_ci(exact=info.exactSolution)
