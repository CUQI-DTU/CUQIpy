#%%
import sys
sys.path.append("../")
import time
import numpy as np
import matplotlib.pyplot as plt

# myfuns
import cuqi
from cuqi.sampler import pCN, MH
from cuqi.distribution import Gaussian, Posterior, DistributionGallery
from cuqi.samples import Samples
#


test_problem = cuqi.testproblem._Deblur()
norm_f = np.linalg.norm(test_problem.exactSolution)

# RHS: measured data
b = test_problem.data
# model
A = test_problem.model
# likelihood
L = test_problem.likelihood

dim = test_problem.model.domain_dim
#prior = cuqi.distribution.GMRF(np.zeros(128), 25, "zero")
prior = cuqi.distribution.Gaussian(np.zeros(dim), 0.2**2)
posterior = cuqi.distribution.Posterior(L, prior)

MCMC_MH = MH(posterior, scale=0.31)
MH_samples = MCMC_MH.sample_adapt(10000,100)
#%%
plt.figure()
MH_samples.plot_mean()
MH_samples.plot_ci(95,exact=test_problem.exactSolution)


# %%
