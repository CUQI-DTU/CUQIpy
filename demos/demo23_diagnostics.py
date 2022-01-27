# %%
import sys
sys.path.append("..")
import cuqi

# %%
dist = cuqi.distribution.DistributionGallery("CalSom91")
sampler = cuqi.sampler.MetropolisHastings(dist)
samples = sampler.sample_adapt(10000)

# %%
# Switch to discrete geometry (easiest for "variable" names)
samples.geometry = cuqi.geometry.Discrete(["alpha","beta"])
ax = samples.plot_autocorrelation()

# %%
samples.burnthin(1000,5).plot_autocorrelation([0],max_lag=60,textsize=25)
