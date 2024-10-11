# %% Initialize and import CUQI
import sys
sys.path.append("..") 
import numpy as np
import cuqi
from cuqi.utilities import plot_2D_density
import matplotlib.pyplot as plt

# %%
# This snippet demonstrates the use of bounded priors.
# Here we use MALA and the truncated normal distribution.
p = cuqi.distribution.TruncatedNormal(
    mean=np.array([0, 0]),
    std=np.array([1, 1]),
    low=np.array([0, -np.Inf]),
    high=np.array([np.Inf, 0]))

plt.figure()
plot_2D_density(p, -5, 5, -5, 5)

sampler = cuqi.experimental.mcmc.MALA(p, scale=0.1, initial_point= np.array([1,-1]))
sampler.sample(10000)
samples = sampler.get_samples()
plt.figure()
samples.plot_trace()
# %%
# This snippet demonstrates the use of bounded priors in solving the simplest
# BIP. Here again we use MALA and the truncated normal prior.
np.random.seed(0)
# the forward model
A_matrix = np.array([[1.0, 1.0]])
A = cuqi.model.LinearModel(A_matrix)
x = cuqi.distribution.TruncatedNormal(
    mean=np.array([0, 0]),
    std=np.array([1, 1]),
    low=np.array([0, -np.Inf]),
    high=np.array([np.Inf, 0]))
# the data distribution
b = cuqi.distribution.Gaussian(A@x, 0.1)

# the observed data
particular_x = np.array([1.5, 1.5])
b_given_particular_x = b(x=particular_x)
b_obs = b_given_particular_x.sample()
print(b_obs)

# the posterior
joint = cuqi.distribution.JointDistribution(x, b)
post = joint(b=b_obs)

sampler = cuqi.experimental.mcmc.MALA(post, initial_point=np.array([2.5, -2.5]), scale=0.03)

sampler.warmup(1000)
sampler.sample(1000)
samples = sampler.get_samples().burnthin(1000)
samples.plot_trace()

# plot exact posterior distribution and samples
# the posterior PDF
plt.figure()
plot_2D_density(post, 1, 5, -3, 1)
plt.title("Exact Posterior")

# samples
plt.figure()
samples.plot_pair()
plt.xlim(1, 5)
plt.ylim(-3, 1)
plt.gca().set_aspect('equal')
plt.title("Posterior Samples")
