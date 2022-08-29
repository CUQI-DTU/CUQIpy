"""
Technical details of samplers
=============================

    This shows technical aspects of the base Sampler class.

"""
# %%
# Setup
# -----

import sys; sys.path.append("../..")
import numpy as np
import cuqi

# %%
# Samplers can be initialized without a target distribution

sampler = cuqi.sampler.MetropolisHastings(scale=0.1)

print(sampler.target)
print(sampler.dim)
print(sampler.geometry)
print(sampler.proposal)
print(sampler.x0)

# %%
# The target can be set later
sampler.target = cuqi.distribution.DistributionGallery('CalSom91')

print(sampler.target)
print(sampler.dim)
print(sampler.geometry)
print(sampler.proposal)
print(sampler.x0)
print(sampler.scale)

sampler.sample(200)

# %%
# We can set up a Gibbs sampler with an initialized sampler

# Model and data
A, y_obs, probinfo = cuqi.testproblem.Deconvolution1D.get_components(phantom='square')

# Get dimension of signal
n = A.domain_dim

# Define distributions
d = cuqi.distribution.Gamma(1, 1e-4)
l = cuqi.distribution.Gamma(1, 1e-4)
x = cuqi.distribution.GMRF(np.zeros(n), lambda d: d[0])
y = cuqi.distribution.GaussianCov(A, lambda l: 1/l)

# Combine into a joint distribution
joint = cuqi.distribution.JointDistribution([d, l, x, y])

# Define posterior by conditioning on the data
posterior = joint(y=y_obs)

# Define sampling strategy
sampling_strategy = {
    'x': cuqi.sampler.Linear_RTO,
    'd': cuqi.sampler.MetropolisHastings(scale=0.1),
    'l': cuqi.sampler.MetropolisHastings(scale=0.1)
}

# Define Gibbs sampler
sampler = cuqi.sampler.Gibbs(posterior, sampling_strategy)

# Run sampler
samples = sampler.sample(Ns=1000, Nb=200)
# %%
samples["d"].plot_trace()
samples["l"].plot_trace()
# %%
