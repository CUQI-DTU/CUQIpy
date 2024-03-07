# %%
from cuqi.distribution import DistributionGallery
from cuqi.mcmc import MHNew

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
sampler.sample(200)

# %%
# Getting samples
#
# To get samples one does as follows:

samples = sampler.get_samples()




# %%
