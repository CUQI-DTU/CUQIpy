
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
R = np.eye(d)
dist = Gaussian(mean= lambda x:x, std=sigma, corrmat = R)
def target(x): return dist.logpdf(x)
ref = Gaussian(mu, np.ones(d), R)
scale = 0.1
x0 = 0.5*np.ones(d)
posterior = cuqi.distribution.Posterior(dist,ref,np.zeros(d))

np.random.seed(0)
MCMC = pCN(posterior, scale, x0)
results1 = MCMC.sample(10,2)

#%%
np.random.seed(0)
likelihood = cuqi.distribution.UserDefinedDistribution(logpdf_func=posterior.loglikelihood_function)
prior = cuqi.distribution.UserDefinedDistribution(sample_func= posterior.prior.sample)

MCMC = pCN((prior,likelihood), scale, x0)
results2 = MCMC.sample(10,2)
# %%
