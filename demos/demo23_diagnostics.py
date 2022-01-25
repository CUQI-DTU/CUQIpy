# %%
import sys
sys.path.append("..")
import cuqi
import arviz as az

%load_ext autoreload
%autoreload 2
# %%
dist = cuqi.distribution.DistributionGallery("CalSom91")
sampler = cuqi.sampler.MetropolisHastings(dist)
samples = sampler.sample_adapt(10000)

# %%
# Switch to discrete geometry (easiest for "variable" names)
samples.geometry = cuqi.geometry.Discrete(["alpha","beta"])
samples.plot_autocorrelation()
# %%
samples.burnthin(1000,5).plot_autocorrelation()
