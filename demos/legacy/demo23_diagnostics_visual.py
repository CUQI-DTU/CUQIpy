# %%
import sys
sys.path.append("..")
import cuqi
# %% 2-dimensional distribution with circular shape
dist = cuqi.distribution.DistributionGallery("CalSom91")
sampler = cuqi.sampler.MH(dist)
samples = sampler.sample_adapt(50000)

# Switch to discrete geometry (easiest for "variable" names)
samples.geometry = cuqi.geometry.Discrete(["alpha","beta"])
# %% traceplot
ax_t = samples.plot_trace()
# %% autocorrelation
ax_ac = samples.plot_autocorrelation()
# %% plot pair-wise scatter plot
ax_p = samples.plot_pair(marginals=True)
# %%
samples_bt = samples.burnthin(20000,20)
samples_bt.plot_autocorrelation()
# %%
samples_bt.plot_trace()
# %% plot marginals
samples_bt.plot_pair(marginals=True)
# %% Try the plotting tools with 4 variable distribution
import numpy as np
#dist4 = cuqi.distribution.Gaussian(np.array([1,2,3,4]),1)
dist4 = cuqi.distribution.LMRF(0, 1, 'zero', geometry=6)
sampler4 = cuqi.sampler.MH(dist4)
samples4 = sampler4.sample_adapt(50000)
samples4.geometry = cuqi.geometry.Discrete(["a","b","c","d","e","f"])
# %%
samples4.plot_trace()
samples4_bt = samples4.burnthin(20000,5)
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
ax_22 = samples4_bt.plot_autocorrelation(np.arange(6), grid=(3,2))
