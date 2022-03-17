#%%
import sys
sys.path.append("..")
import numpy as np
import matplotlib.pyplot as plt
import cuqi

#%% Load testproblem
TP = cuqi.testproblem.Deconv_2D() #Default values

# %% Add prior (using the geometry of the deconvolution model)
TP.prior = cuqi.distribution.GaussianCov(mean=np.zeros(TP.model.domain_dim),
                                         cov=1,
                                         geometry=TP.model.domain_geometry)

#%% Plot data
TP.data.plot()

#%% Plot noise free data
TP.exactData.plot()

#%% Plot noise relalization (only possible since we know noise free data for test case)
(TP.data-TP.exactData).plot()

#%% Plot exact solution (phantom)
TP.exactSolution.plot()

#%% Plot samples from prior
TP.prior.sample(5).plot()

#%% Now sample posterior
post_samples = TP.sample_posterior(200)

#%% Plot mean of posterior
post_samples.plot_mean()

# %% Plot samples from posterior
post_samples.plot()

#%% Plot "Credibility interval". Should create a number of plots for 2D
post_samples.plot_ci(95)
