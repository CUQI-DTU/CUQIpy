# %% Initialize and import CUQI
import sys
sys.path.append("..") 
import numpy as np
import cuqi
from cuqi.utilities import plot_2D_density
import matplotlib.pyplot as plt

# %%
# This snippet demonstrates how to draw samples from bounded priors with MALA.
bounded_prior_lists = [
    cuqi.distribution.Beta(np.array([2,2]),np.array([2,2])),
    cuqi.distribution.Uniform(np.array([0,0]),np.array([1,1])),
    cuqi.distribution.TruncatedNormal(np.array([0,0]),np.array([1,1]),
                                      np.array([-2,-2]),np.array([2,2]))
    ]

for dist in bounded_prior_lists:
    sampler = cuqi.experimental.mcmc.MALA(dist, scale=0.1)
    samples = dist.sample(1000)
    samples.plot_pair()
    plt.title("Samples of {}".format(dist.__class__.__name__))
    plt.show()
# %%
# This snippet demonstrates how to solve the simplest BIP with bounded priors.
