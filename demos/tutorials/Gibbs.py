"""
Gibbs sampling
==============

    This tutorial shows how to use cuqipy to perform Gibbs sampling.

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
# We define the following model:
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

# Define distributions
d = Gamma(1, 1e-4)
l = Gamma(1, 1e-4)
x = GMRF(np.zeros(A.domain_dim), lambda d: d[0])
#x = GaussianCov(np.zeros(A.domain_dim), lambda d: 1/d)
y = GaussianCov(A, lambda l: 1/l)

# We now combine these distributions into a joint distribution
joint = JointDistribution([d, l, x, y])

# And define our posterior by conditioning the joint on the data
posterior = joint(y=y_obs)

print(posterior)

# %%
# Gibbs Sampler
# -------------
# We can sample from the posterior using Gibbs sampling.
# We first create a dictionary of samplers for each parameter.
# We use the Linear_RTO sampler for the :math:`\mathbf{x}` parameter,
# and the Conjugate sampler for the other parameters.

# Define Gibbs sampler
sampler = Gibbs(posterior,{'x': Linear_RTO, ('d', 'l'): Conjugate})

# Run sampler
samples = sampler.sample(Ns=500, Nb=200)

# %%
samples['x'].plot_ci(exact=probinfo.exactSolution)
samples['d'].plot_trace()
samples['l'].plot_trace()

# %%
