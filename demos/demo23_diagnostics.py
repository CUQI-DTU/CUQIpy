# %%
import sys
sys.path.append("..")
import cuqi
import numpy as np
# %% 2-dimensional distribution with circular shape
dist = cuqi.distribution.DistributionGallery("CalSom91")
sampler1 = cuqi.sampler.MH(dist)
sampler2 = cuqi.sampler.MH(dist)
sampler1.warmup(50000)
samples1 = sampler1.get_samples()
sampler2.warmup(50000)
samples2 = sampler2.get_samples()
# %%
# Switch to discrete geometry (easiest for "variable" names)
samples1.geometry = cuqi.geometry.Discrete(["alpha","beta"])
samples2.geometry = cuqi.geometry.Discrete(["alpha","beta"])

# %% ESS
print(f"samples1: {samples1.compute_ess()}")
print(f"samples2: {samples2.compute_ess()}")
# %% RHAT
print(f"rhat values: {samples1.compute_rhat(samples2)}")
