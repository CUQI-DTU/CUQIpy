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
ax_t = samples.plot_trace()
# %% autocorrelation
ax_ac = samples.plot_autocorrelation()
# %% plot marginals
ax_p = samples.plot_pair()
# %%
samples_bt = samples.burnthin(20000,20)
samples_bt.plot_autocorrelation()
# %%
samples_bt.plot_trace()
# %% plot marginals
samples_bt.plot_pair()
# %% Try all plotting tools with 4 variables
import numpy as np
dist4 = cuqi.distribution.Gaussian(np.array([1,2,3,4]),1)
sampler4 = cuqi.sampler.MetropolisHastings(dist4)
samples4 = sampler4.sample_adapt(20000)
samples4.geometry = cuqi.geometry.Discrete(["a","b","c","d"])
# %%
samples4.plot_trace()
samples4_bt = samples4.burnthin(10000)
# %%
samples4_bt.plot_trace()
# %%
samples4_bt.plot_pair(textsize=25)
# %%
samples4_bt.plot_pair(marginals=True, textsize=25)
# %%
samples4_bt.plot_pair(kind='kde', textsize=25)
# %%
samples4_bt.plot_pair(kind='hexbin', textsize=25)
# %%
ax_4 = samples4_bt.plot_autocorrelation()
# %%
ax_22 = samples4_bt.plot_autocorrelation(label_axis=False, grid=(2,2))