"""
Samplers
========

This tutorial will cover the basic concepts of samplers in CUQIpy.

"""
# %%
import cuqi
import numpy as np
import time

# %%
#
# CUQIpy provides a number of samplers, which can be used to sample from a given
# target distribution. The samplers are implemented as classes, which can be
# instantiated and used to sample from a target distribution.

target = cuqi.distribution.Gaussian(np.zeros(2), 1)

sampler = cuqi.sampler.MH(target)
sampler_old = cuqi.sampler.MetropolisHastings(target)

# %%
tic = time.time()
sampler_old.sample_adapt(1000)
print(f"Time elapsed: {time.time() - tic:.2f} s")

# %%
tic = time.time()
sampler.warmup(1000)
print(f"Time elapsed: {time.time() - tic:.2f} s")


# %%

sampler.get_status()

# %%
