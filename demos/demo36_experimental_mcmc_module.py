# %%
from cuqi.distribution import DistributionGallery
from cuqi.experimental.mcmc import MHNew
# %%
# The samplers in the MCMC module are an re-implementation of this sampler module in a more object oriented way.
# The sampling results are tested against the samplers in the sampler module in test_mcmc.py.
#

# Define posterior target to illustrate one

target = DistributionGallery("BivariateGaussian")


# %%
# Basic sampling
#
# Basic sampling now works via the combining "warmup" and "sample" as follows.
# Illustrated here on MH sampler

# %%
# Initialize sampler (including optional sampler parameters)
sampler = MHNew(target, scale=0.1)

# %%
# Warmup sampler. This tunes the internal parameters (scale here) using the `.tune` at internals defined by optional tune_freq parameter
sampler.warmup(200)
print(sampler.scale) # Scale should be changed

# %%
# Sampling. This runs the sampler with fixed internal parameters.
sampler.sample(500)
print(sampler.scale) # Should be the same

# %%
# Warmup and sample can be combined in arbitrary ways e.g.:
sampler.sample(500)
sampler.warmup(100)
sampler.warmup(100)
sampler.sample(500)

# %%
# Getting samples
#
# To get samples one does as follows:

samples = sampler.get_samples()
print(samples)
print(f"Shape: {samples.shape}")

# Initial point, and all samples (both warmup and sample) are stored
# It is currently up to the user to remove burn-in after sampling, e.g.
samples.burnthin(1401).plot_pair()

# %%
# Checkpointing
#
# The new sampler supports checkpointing by saving and loading state (not samples) For example:

# Save current state of sampler (e.g. current point, current scale etc.)
sampler.save_checkpoint('demo36_sampler_checkpoint.pickle')

# Then using a new sampler can load checkpoint
sampler2 = MHNew(target)
sampler2.load_checkpoint('demo36_sampler_checkpoint.pickle')

print(sampler.scale) # Should be the same
print(sampler2.scale) # Should be the same

# %%
# Batch sampling
#
# The new sampler supports batch sampling. This is useful for saving samples to disk in chunks
# while sampling. For example:

sampler.sample(500, batch_size=100, sample_path='demo36_sampler_samples/')

# This stored samples in chunks of 100 in the folder demo36_sampler_samples
