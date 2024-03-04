# %%
import numpy as np
import cuqi
from cuqi.sampler import Sampler
import matplotlib.pyplot as plt
from cuqi.array import CUQIarray
#%%
# Define custom distribution
# this is from tests/test_pCN_sample_regression.py
def make_custom_distribution(d):
    np.random.seed(0)
    mu = np.zeros(d)
    sigma = np.linspace(0.5, 1, d)
    model = cuqi.model.Model(lambda x: x, range_geometry=d, domain_geometry=d)
    L = cuqi.distribution.Gaussian(mean=model, sqrtcov=sigma).to_likelihood(np.zeros(d))
    def target(x): return L.logd(x)
    P = cuqi.distribution.Gaussian(mu, np.ones(d))
    target = cuqi.distribution.Posterior(L, P)
    return target

dim = 2
target = make_custom_distribution(dim)
scale = 0.1
x0 = 0.5*np.ones(dim)

np.random.seed(0)

MCMC = cuqi.sampler.pCN(target, scale, x0)
results = MCMC.sample(1000)
results.plot_chain()
plt.title("old PCN")
# %%
np.random.seed(0)
sampler = cuqi.mcmc.PCN_new(target, scale=scale, initial_point=x0)
# Sample
# TODO: there seems a bug with warmup, but I can't reproduce it
sampler.sample(1000)
samples = sampler.get_samples()
plt.figure()
plt.plot(samples.samples[:,0])
plt.plot(samples.samples[:,1])
plt.title("new PCN")

# %% test checkpointing with the new MALA sampler
np.random.seed(0)
sampler.sample(10000)

sampler.save_checkpoint('checkpoint.pickle')

np.random.seed(0)

sampler.reset()
sampler.sample(1000)

samples = sampler.get_samples()

f, axes = plt.subplots(1,2)

axes[0].plot(samples.samples[:,1])
axes[0].set_title('without checkpoint')

sampler2 = cuqi.mcmc.PCN_new(target, scale=0.1, initial_point=x0)

sampler2.load_checkpoint('checkpoint.pickle')

np.random.seed(0)

sampler2.sample(1000)
axes[1].plot(samples.samples[:,1])
axes[1].set_title('with loaded checkpoint')
plt.show()
