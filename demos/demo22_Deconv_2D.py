#%%
import sys

sys.path.append("..")
import numpy as np
import matplotlib.pyplot as plt
import cuqi

#import skimage
#from skimage import color
#phantom = color.rgb2gray(skimage.data.astronaut())
phantom = "satellite"

#%% %Cuqi deblur test problem
model, data, probInfo = cuqi.testproblem.Deconv_2D.get_components(dim=128, phantom=phantom)
likelihood = cuqi.distribution.GaussianCov(model, 0.05**2).to_likelihood(data)
prior = cuqi.distribution.GaussianCov(np.zeros(model.domain_dim), 1, geometry=model.domain_geometry)
BP = cuqi.problem.BayesianProblem(likelihood, prior)

#%%
data.plot()

#%%
probInfo.exactData.plot()

#%%
(data-probInfo.exactData).plot()

#%%
probInfo.exactSolution.plot()

#%%
prior.sample(5).plot()

#%%
post_samples = BP.sample_posterior(200)

#%%
post_samples.plot_mean()

#%%
post_samples.plot_ci(95)
