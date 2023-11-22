# %%
import sys
sys.path.append("..")
import cuqi
import numpy as np
import matplotlib.pyplot as plt

# %%
TP = cuqi.testproblem.Deconvolution1D()
TP.prior=cuqi.distribution.Gaussian(np.zeros(TP.model.range_dim),0.1)
#%%
samples = TP.sample_posterior(500)
# %%

samples.plot_mean() # Should compute mean and plot according to the geometry of the object


# %%
samples.plot()

# %%
samples.plot_chain(39)

# %%
samples.plot_ci(95)
# %%
