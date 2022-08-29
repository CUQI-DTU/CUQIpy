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

# %%
# We can set up a Gibbs sampler with an initialized sampler
# Here we choose a very sample 2D Gaussian to Gibbs sample from

# Define distributions
x = cuqi.distribution.GaussianCov(0, 1)
y = cuqi.distribution.GaussianCov(5, 1)

# Combine into a joint distribution
target = cuqi.distribution.JointDistribution([x, y])

# Define sampling strategy
sampling_strategy = {
    'x': cuqi.sampler.MetropolisHastings(scale=0.1),
    'y': cuqi.sampler.MetropolisHastings(scale=0.1),
}

# Define Gibbs sampler
sampler = cuqi.sampler.Gibbs(target, sampling_strategy)

# Run sampler
samples = sampler.sample(Ns=1000, Nb=200)
# %%
# Plot traces
samples["x"].plot_trace()
samples["y"].plot_trace()

# %%
# Try sampling x directly
sampler_x = cuqi.sampler.MetropolisHastings(target=x, scale=0.1)
samples_x = sampler_x.sample_adapt(N=1000, Nb=200)
samples_x.plot_trace()

# Try sampling y directly
sampler_y = cuqi.sampler.MetropolisHastings(target=y, scale=0.1)
samples_y = sampler_y.sample_adapt(N=1000, Nb=200)
samples_y.plot_trace()
# %%
