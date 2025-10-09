# %% Initialize and import CUQIpy
import sys
sys.path.append("..") 
import numpy as np
import cuqi
from cuqi.utilities import plot_2D_density
import matplotlib.pyplot as plt

# %% Demo 1
# This demo shows the use of gradient-based MCMC methods in drawing samples 
# from a truncated normal distribution. Here we use MALA.
p = cuqi.distribution.TruncatedNormal(
    mean=np.array([0, 0]),
    std=np.array([1, 1]),
    low=np.array([0, -np.inf]),
    high=np.array([np.inf, 0]))

plt.figure()
plot_2D_density(p, -5, 5, -5, 5)
plt.title("Exact PDF")

sampler = cuqi.sampler.MALA(p, scale=0.1, initial_point= np.array([1,-1]))
sampler.sample(10000)
samples = sampler.get_samples()

plt.figure()
samples.plot_pair()
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.gca().set_aspect('equal')
plt.title("Samples")

# %% Demo 2
# This demo shows the use of truncted normal as prior in solving the simplest
# BIP. Here again we use MALA.
np.random.seed(0)
# the forward model
A_matrix = np.array([[1.0, 1.0]])
A = cuqi.model.LinearModel(A_matrix)
x = cuqi.distribution.TruncatedNormal(
    mean=np.array([0, 0]),
    std=np.array([1, 1]),
    low=np.array([0, -np.inf]),
    high=np.array([np.inf, 0]))
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

# MALA sampler
sampler = cuqi.sampler.MALA(post, initial_point=np.array([2.5, -2.5]), scale=0.03)

sampler.warmup(1000)
sampler.sample(1000)
samples = sampler.get_samples().burnthin(1000)
samples.plot_trace()

# plot exact posterior distribution and samples
# the posterior PDF
plt.figure()
plot_2D_density(post, 1, 5, -3, 1)
plt.title("Exact Posterior PDF")

# samples
plt.figure()
samples.plot_pair()
plt.xlim(1, 5)
plt.ylim(-3, 1)
plt.gca().set_aspect('equal')
plt.title("Samples")
