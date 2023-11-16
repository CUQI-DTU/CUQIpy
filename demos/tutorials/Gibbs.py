"""
Gibbs sampling
==============

    This tutorial shows how to use CUQIpy to perform Gibbs sampling.
    Gibbs sampling is a Markov chain Monte Carlo (MCMC) method for
    sampling a joint probability distribution.

    Opposed to jointly sampling the distribution simultaneously, Gibbs
    sampling samples the variables of the distribution sequentially,
    one variable at a time. When a variable represents a random vector, the
    whole vector is sampled simultaneously.
    
    The sampling of each variable is done by sampling from the conditional
    distribution of that variable given (fixed, previously sampled) values
    of the other variables.

    This is often a very efficient way of sampling from a joint
    distribution if the conditional distributions are easy to sample
    from. This is one way to exploit the structure of the joint
    distribution. On the other hand, if the conditional distributions
    are highly correlated and/or are difficult to sample from, then
    Gibbs sampling can be very inefficient.

    For these reasons, Gibbs sampling is often a double-edged sword,
    that needs to be used in the right context.

"""
# %%
# Setup
# -----
# We start by importing the necessary modules

import numpy as np
import matplotlib.pyplot as plt
from cuqi.testproblem import Deconvolution1D
from cuqi.distribution import Gaussian, Gamma, JointDistribution, GMRF, LMRF
from cuqi.sampler import Gibbs, LinearRTO, Conjugate, UGLA, ConjugateApprox

np.random.seed(0)

# %%
# Forward model and data
# ------------------------
# We define the forward model and data.
# Here we use a 1D deconvolution problem, so the forward model is linear,
# that is:
#
# .. math::
#    \mathbf{y} = \mathbf{A} \mathbf{x}
#
# where :math:`\mathbf{A}` is the convolution matrix, and :math:`\mathbf{x}` is the input signal.
#
# We load this example from the testproblem library of CUQIpy and visualize the
# true solution (sharp signal) and data (convolved signal).

# Model and data
A, y_obs, probinfo = Deconvolution1D(phantom='square').get_components()

# Get dimension of signal
n = A.domain_dim

# Plot exact solution and observed data
plt.subplot(121)
probinfo.exactSolution.plot()
plt.title('exact solution')

plt.subplot(122)
y_obs.plot()
plt.title("Observed data")

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
#         \mathbf{x} &\sim \mathrm{GMRF}(\mathbf{0}, d) \\
#         \mathbf{y} &\sim \mathcal{N}(\mathbf{A} \mathbf{x}, l^{-1} \mathbf{I}_m)
#     \end{align}
#
# where :math:`\mathbf{y}` is the observed data, and :math:`\mathbf{x}`
# is the unknown signal. The hyperparameters :math:`d` and :math:`l` are
# the precision of the prior distribution of :math:`\mathbf{x}` and
# the noise, respectively.
#
# The prior distribution of :math:`\mathbf{x}` is a Gaussian Markov random
# field (GMRF) with zero mean and precision :math:`d`. It can
# be viewed as a Gaussian prior on the differences between neighboring
# elements of :math:`\mathbf{x}`.
#
# In CUQIpy the model can be defined as follows:

# Define distributions
d = Gamma(1, 1e-4)
l = Gamma(1, 1e-4)
x = GMRF(np.zeros(n), lambda d: d)
y = Gaussian(A, lambda l: 1/l)

# Combine into a joint distribution
joint = JointDistribution(d, l, x, y)

# View the joint distribution
print(joint)

# %%
# Notice that the joint distribution prints a mathematical expression
# for the density functions that make up :math:`p(d,l,\mathbf{x},\mathbf{y})`.
# In this case they are all distributions, but this need not be the case.

# %%
# Defining the posterior distribution
# ------------------------------------
#
# Now we define the posterior distribution, which is the joint distribution
# conditioned on the observed data. That is, :math:`p(d, l, \mathbf{x} \mid \mathbf{y}=\mathbf{y}_\mathrm{obs})`
#
# This is done in the following way:

# Define posterior by conditioning on the data
posterior = joint(y=y_obs)

# View the structure of the posterior
print(posterior)

# %%
# Notice that after conditioning on the data, the distribution associated with
# :math:`\mathbf{y}` became a likelihood function and that the posterior is now
# a joint distribution of the variables :math:`d`, :math:`l`, :math:`\mathbf{x}`.

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
# variables using the ``LinearRTO`` sampler.
# 
# Taking these two facts into account, we can define a Gibbs sampler
# that uses the ``Conjugate`` sampler for :math:`d` and :math:`l` and
# the ``LinearRTO`` sampler for :math:`\mathbf{x}`.
#
# This is done in CUQIpy as follows:

# Define sampling strategy
sampling_strategy = {
    'x': LinearRTO,
    'd': Conjugate,
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
# After sampling we can inspect the results. The samples are stored
# as a dictionary with the variable names as keys. Samples for each 
# variable is stored as a CUQIpy Samples object which contains the
# many convenience methods for diagnostics and plotting of MCMC samples.

# Plot credible intervals for the signal
samples['x'].plot_ci(exact=probinfo.exactSolution)

# %%
# Trace plot for d
samples['d'].plot_trace(figsize=(8,2))

# %%
# Trace plot for l
samples['l'].plot_trace(figsize=(8,2))

# %%
# Switching to a piecewise constant prior
# ---------------------------------------
#
# Notice that while the sampling went well in the previous example,
# the posterior distribution did not match the characteristics of
# the exact solution. We can improve this result by switching to a
# prior that better matches the exact solution :math:`\mathbf{x}`.
#
# One choice is the Laplace difference prior, which assumes a
# Laplace distribution for the differences between neighboring
# elements of :math:`\mathbf{x}`. That is,
#
# .. math::
#
#     \mathbf{x} \sim \text{LMRF}(d^{-1}),
#
# which means that :math:`x_i-x_{i-1} \sim \mathrm{Laplace}(0, d^{-1})`.
#
# This prior is implemented in CUQIpy as the ``LMRF`` distribution.
# To update our model we simply need to replace the ``GMRF`` distribution
# with the ``LMRF`` distribution. Note that the Laplace distribution
# is defined via a scale parameter, so we invert the parameter :math:`d`.
#
# This laplace distribution and new posterior can be defined as follows:

# Define new distribution for x
x = LMRF(0, lambda d: 1/d, geometry=n)

# Define new joint distribution with piecewise constant prior
joint_Ld = JointDistribution(d, l, x, y)

# Define new posterior by conditioning on the data
posterior_Ld = joint_Ld(y=y_obs)

print(posterior_Ld)


# %%
# Gibbs Sampler (with Laplace prior)
# ----------------------------------
#
# Using the same approach as earlier we can define a Gibbs sampler
# for this new hierarchical model. The only difference is that we
# now need to use a different sampler for :math:`\mathbf{x}` because
# the ``LinearRTO`` sampler only works for Gaussian distributions.
#
# In this case we use the UGLA (Unadjusted Gaussian Laplace Approximation) sampler
# for :math:`\mathbf{x}`. We also use an approximate Conjugate
# sampler for :math:`d` which approximately samples from the
# posterior distribution of :math:`d` conditional on the other
# variables in an efficient manner. For more details see e.g.
# `this paper <https://arxiv.org/abs/2104.06919>`.

# Define sampling strategy
sampling_strategy = {
    'x': UGLA,
    'd': ConjugateApprox,
    'l': Conjugate
}

# Define Gibbs sampler
sampler_Ld = Gibbs(posterior_Ld, sampling_strategy)

# Run sampler
samples_Ld = sampler_Ld.sample(Ns=1000, Nb=200)

# %%
# Analyze results
# ---------------
#
# Again we can inspect the results.
# Here we notice the posterior distribution matches the exact solution much better.

# Plot credible intervals for the signal
samples_Ld['x'].plot_ci(exact=probinfo.exactSolution)
#%%
samples_Ld['d'].plot_trace(figsize=(8,2))
#%%
samples_Ld['l'].plot_trace(figsize=(8,2))
