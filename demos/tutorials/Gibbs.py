"""
Gibbs sampling
==============

    This tutorial shows how to use cuqipy to perform Gibbs sampling.
    Gibbs sampling is a Markov chain Monte Carlo (MCMC) method for
    sampling a joint probability distribution.

    Opposed to jointly sampling the distribution simultaneously, Gibbs
    sampling samples the variables of the distribution sequentially,
    one variable at a time. The sampling of each variable is done by
    sampling from the conditional distribution of that variable given 
    the values of the other variables.

    This is often a very efficient way of sampling from a joint
    distribution if the conditional distributions are easy to sample
    from. This is one way to exploit the structure of the joint
    distribution.

"""
# %%
# Setup
# -----
# We start by importing the necessary modules and loading the model and data.
# The model is a simple 1D convolution, but any model can be used.
# We also define some helper variables to make the code more readable.

import numpy as np
from cuqi.testproblem import Deconvolution1D
from cuqi.distribution import GaussianCov, Gamma, JointDistribution, GMRF, Laplace_diff
from cuqi.sampler import Gibbs, Linear_RTO, Conjugate, UnadjustedLaplaceApproximation, ConjugateApprox

# Model and data
A, y_obs, probinfo = Deconvolution1D.get_components(phantom='square')

# %%
# Hierarchical Bayesian model
# ---------------------------
#
# We define the following hierarchical model:
#
# .. math::
#
#     \begin{align}
#         d &\sim \mathrm{Gamma}(1, 10^{-4}) \\
#         l &\sim \mathrm{Gamma}(1, 10^{-4}) \\
#         \mathbf{x} &\sim \mathrm{GMRF}(\mathbf{0}, d^2 \mathbf{I}_n) \\
#         \mathbf{y} &\sim \mathcal{N}(\mathbf{A} \mathbf{x}, l^2 \mathbf{I}_m)
#     \end{align}
#
# where :math:`\mathbf{A}` is the convolution matrix, :math:`\mathbf{y}` is
# the observed data, and :math:`\mathbf{x}` is the unknown signal.
# The hyperparameters :math:`d` and :math:`l` are the precision of the
# prior distribution of :math:`\mathbf{x}` and the noise, respectively.
#
# The prior distribution of :math:`\mathbf{x}` is a Gaussian Markov random
# field (GMRF) with zero mean and precision :math:`d^2 \mathbf{I}_n`. It can
# be viewed as a Gaussian prior on the differences between neighboring
# elements of :math:`\mathbf{x}`.
#
# In CUQIpy the model can be defined as follows:

# Define distributions
d = Gamma(1, 1e-4)
l = Gamma(1, 1e-4)
x = GMRF(np.zeros(A.domain_dim), lambda d: d[0])
y = GaussianCov(A, lambda l: 1/l)

# Combine into a joint distribution
joint = JointDistribution([d, l, x, y])

# Define posterior by conditioning on the data
posterior = joint(y=y_obs)

print(posterior)

# %%
# Gibbs Sampler
# -------------
#
# The hierarchical model above has some important properties that we
# can exploit to make the sampling more efficient. First, note that
# the Gamma distribution are conjugate priors for the precision of
# the Gaussian distributions. This means that we can efficiently sample
# from :math:`d` and :math:`l` conditional on the other variables.
# 
# Second, note that the prior distribution of :math:`\mathbf{x}` is
# a Gaussian Markov random field (GMRF) and that the distribution for
# :math:`\mathbf{y}` is also Gaussian with a Linear operator acting
# on :math:`\mathbf{x}` as the mean variable. This means that we can
# efficiently sample from :math:`\mathbf{x}` conditional on the other
# variables using the ``Linear_RTO`` sampler.
# 
# Taking these two facts into account, we can define a Gibbs sampler
# that uses the ``Conjugate`` sampler for :math:`d` and :math:`l` and
# the ``Linear_RTO`` sampler for :math:`\mathbf{x}`.
#
# This is done in CUQIpy as follows:

# Define Gibbs sampler
sampler = Gibbs(posterior, {'x': Linear_RTO, ('d', 'l'): Conjugate})

# Run sampler
samples = sampler.sample(Ns=1000, Nb=200)

# %%
# Analyze results
# ---------------
#
# After sampling we can inspect the results. The samples are stored
# as a dictionary with the variable names as keys. Each variable is
# stored as a CUQIpy Samples object which contains the many convenience
# methods for diagnostics and plotting of MCMC samples.

# Plot credible intervals for the signal and trace plots for hyperparameters
samples['x'].plot_ci(exact=probinfo.exactSolution)
samples['d'].plot_trace()
samples['l'].plot_trace()

# %%
# Switching to a piecewise constant prior
# ---------------------------------------
#
# Notice that while the sampling went well in the previous example,
# the posterior distribution did not match the characteristics of
# the exact solution. We can improve this result by switching to a
# piecewise constant prior for :math:`\mathbf{x}`.
#
# One choice is the Laplace difference prior, which assumes a
# Laplace distribution for the differences between neighboring
# elements of :math:`\mathbf{x}`. That is, we assume that
#
# .. math::
#
#     \mathbf{x}_i - \mathbf{x}_{i-1} \sim \mathrm{Laplace}(0, d)
#
# This prior is implemented in CUQIpy as the ``Laplace_diff`` distribution.
# To update our model we simply need to replace the ``GMRF`` distribution
# with the ``Laplace_diff`` distribution. Note that the Laplace distribution
# is defined via a scale parameter, so we invert the parameter :math:`d`.
#
# This laplace distribution and new posterior can be defined as follows:

# Define new distribution for x
x = Laplace_diff(np.zeros(A.domain_dim), lambda d: 1/d)

# Define new joint distribution
joint = JointDistribution([d, l, x, y])

# Define new posterior by conditioning on the data
posterior = joint(y=y_obs)

print(posterior)


# %%
# Gibbs Sampler (with Laplace prior)
# ----------------------------------
#
# Using the same approach as ealier we can define a Gibbs sampler
# for this new hierarchical model. The only difference is that we
# now need to use a different sampler for :math:`\mathbf{x}` because
# the ``Linear_RTO`` sampler only works for Gaussian distributions.
#
# In this case we use the UnadjustedLaplaceApproximation sampler
# for :math:`\mathbf{x}`. We also use an approximate Conjugate
# sampler for :math:`d` which approximately samples from the
# posterior distribution of :math:`d` conditional on the other
# variables in an efficient manner. For more details see e.g.
# `this paper <https://arxiv.org/abs/2104.06919>`.

# Define sampling strategy
sampling_strategy = {
    'x': UnadjustedLaplaceApproximation,
    'd': ConjugateApprox,
    'l': Conjugate
}

# Define Gibbs sampler
sampler = Gibbs(posterior, sampling_strategy)

# Run sampler
samples = sampler.sample(Ns=1000, Nb=200)

# %%
# Analyze results
# ---------------
#
# Again we can inspect the results.
# Here we notice the posterior distribution matches the exact solution much better.

# Plot credible intervals for the signal and trace plots for hyperparameters
samples['x'].plot_ci(exact=probinfo.exactSolution)
samples['d'].plot_trace()
samples['l'].plot_trace()
