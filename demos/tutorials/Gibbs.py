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
from cuqi.distribution import GaussianCov, Gamma, JointDistribution, GMRF
from cuqi.sampler import Gibbs, Linear_RTO, Conjugate

# Model and data
A, y_obs, probinfo = Deconvolution1D.get_components(phantom='pc')

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

