
#%%
import sys

from numpy import random
sys.path.append("..")
import cuqi
import numpy as np

from cuqi.distribution import Gaussian
from cuqi.sampler import pCN

#%%
np.random.seed(0)
d= 2
mu = np.zeros(d)
sigma = np.linspace(0.5, 1, d)

model = cuqi.model.Model(lambda x:x, range_geometry=d, domain_geometry=d)
L = Gaussian(model, std=sigma**2).to_likelihood(np.zeros(d))
def target(x): return L.logd(x)
P = Gaussian(mu, np.ones(d))
scale = 0.1
x0 = 0.5*np.ones(d)
posterior = cuqi.distribution.Posterior(L, P)

np.random.seed(0)
MCMC = pCN(posterior, scale, x0)
results1 = MCMC.sample(10,2)

#%%
np.random.seed(0)
likelihood = cuqi.likelihood.UserDefinedLikelihood(dim=d, logpdf_func=posterior.likelihood.logd)
prior = cuqi.distribution.UserDefinedDistribution(sample_func= posterior.prior.sample)

MCMC = pCN((likelihood,prior), scale, x0)
results2 = MCMC.sample(10,2)
# %%
