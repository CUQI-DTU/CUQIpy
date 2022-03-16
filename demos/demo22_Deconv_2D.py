#%%
import sys
import time
sys.path.append("../")

import numpy as np
import matplotlib.pyplot as plt

import cuqi

#%% %Cuqi deblur test problem
tp = cuqi.testproblem.Deconv_2D() #Default values
tp.prior = cuqi.distribution.GaussianCov(np.zeros(tp.model.domain_dim), 1, geometry=tp.model.domain_geometry)
#%%
tp.data.plot()

#%%
tp.exactData.plot()

#%%
(tp.data-tp.exactData).plot()

#%%
tp.exactSolution.plot()

#%%
tp.prior.sample(5).plot()

#%%
post_samples = tp.sample_posterior(200)

#%%
post_samples.plot_mean()

#%%
post_samples.plot_ci(95)
