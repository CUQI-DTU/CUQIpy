# %%
import numpy as np
import cuqi
import matplotlib.pyplot as plt

#%%
# Define custom distribution
def make_custom_distribution(dim=5):
    mu = np.arange(dim)  # Mean of Gaussian
    std = 1  # standard deviation of Gaussian

    # Logpdf function
    logpdf_func = lambda x: -1/(std**2)*np.sum((x-mu)**2)
    gradient_func = lambda x: -2/(std**2)*(x-mu)

    # Define distribution from logpdf as UserDefinedDistribution (sample and gradients also supported)
    target = cuqi.distribution.UserDefinedDistribution(dim=dim, logpdf_func=logpdf_func,
                                                        gradient_func=gradient_func)
    return target

dim = 5
target = make_custom_distribution(dim)
eps = 1/dim
N = 2000
x0 = np.zeros(dim)

# %% Compare old MALA sampler vs new MALA sampler
# Set up old MALA sampler
np.random.seed(0)
sampler_old = cuqi.sampler.MALA(target, scale=eps**2, x0=x0)
# Sample
samples_old = sampler_old.sample(N)
plt.figure()
samples_old.plot_chain()
plt.title('old MALA')
# Set up new MALA sampler
np.random.seed(0)
sampler = cuqi.mcmc.MALANew(target, scale=eps**2, initial_point=x0)
# Sample
sampler.sample(N)
samples = sampler.get_samples()

plt.figure()
plt.plot(samples.samples[:,0])
plt.plot(samples.samples[:,1])
plt.plot(samples.samples[:,2])
plt.plot(samples.samples[:,3])
plt.plot(samples.samples[:,4])
plt.title('new MALA')

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

sampler2 = cuqi.mcmc.MALANew(target, scale=eps**2, initial_point=x0)

sampler2.load_checkpoint('checkpoint.pickle')

np.random.seed(0)

sampler2.sample(1000)
samples2 = sampler2.get_samples()   
axes[1].plot(samples2.samples[:,1])
axes[1].set_title('with loaded checkpoint')
plt.show()

# %%
