# %%
import sys
sys.path.append("..")
import cuqi

# %%
dist = cuqi.distribution.DistributionGallery("CalSom91")
sampler = cuqi.sampler.MetropolisHastings(dist)
samples = sampler.sample_adapt(50000)

# Switch to discrete geometry (easiest for "variable" names)
samples.geometry = cuqi.geometry.Discrete(["alpha","beta"])

# %% traceplot
ax = samples.plot_trace()

# %% autocorrelation
ax = samples.plot_autocorrelation()

# %%
samples_bt = samples.burnthin(20000,20)
samples_bt.plot_autocorrelation()
# %%
samples_bt.plot_trace()
