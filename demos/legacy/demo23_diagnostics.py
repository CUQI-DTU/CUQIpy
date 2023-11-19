# %%
import sys
sys.path.append("..")
import cuqi
import numpy as np
# %% 2-dimensional distribution with circular shape
dist = cuqi.distribution.DistributionGallery("CalSom91")
sampler = cuqi.sampler.MH(dist)
samples1 = sampler.sample_adapt(50000)
samples2 = sampler.sample_adapt(50000)
# %%
# Switch to discrete geometry (easiest for "variable" names)
samples1.geometry = cuqi.geometry.Discrete(["alpha","beta"])
samples2.geometry = cuqi.geometry.Discrete(["alpha","beta"])

# %% ESS
print(f"samples1: {samples1.compute_ess()}")
print(f"samples2: {samples2.compute_ess()}")
# %% RHAT
print(f"rhat values: {samples1.compute_rhat(samples2)}")
