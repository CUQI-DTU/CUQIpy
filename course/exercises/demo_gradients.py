# %%
import sys
sys.path.append("../../")
import time
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.sparse as sps

# myfuns
import cuqi
%load_ext autoreload
%autoreload 2

# %%

prior = cuqi.distribution.GaussianGen(0, 5)
#prior = cuqi.distribution.GaussianGen(np.zeros(1), 5*np.eye(1))
#prior = cuqi.distribution.GaussianGen(np.zeros(5), 5*np.eye(5))
#prior = cuqi.distribution.GaussianGen(np.zeros(5), 5)
#prior = cuqi.distribution.GaussianGen(0,5*np.ones(5))
#prior = cuqi.distribution.GaussianGen(0,5*np.eye(5))
#prior = cuqi.distribution.GaussianGen(0,5*sps.eye(5))
#prior = cuqi.distribution.GaussianGen(np.zeros(5), np.array([1,2,9,5,2]))
#prior = cuqi.distribution.GaussianGen(np.zeros(5001), 5*np.eye(5001))
#prior = cuqi.distribution.GaussianGen(np.zeros(5001), 5*sps.eye(5001))
#prior = cuqi.distribution.GaussianGen(np.zeros(2), sps.csc_matrix([[5,3],[-3,2]]))
#A = 100*np.random.randn(5001,5001)
#prior = cuqi.distribution.GaussianGen(np.zeros(5001), A@A.T)
# %%
# Some basic checkss
x0 = 1000*np.random.rand(prior.dim)
eval1 = prior.logpdf(x0)
eval2 = sp.stats.multivariate_normal.logpdf(x0, 0*np.ones(prior.dim), prior.cov)
print(np.linalg.norm(eval1-eval2)/np.linalg.norm(eval2))
# %%
gradeval1 = prior.gradient(x0)
gradeval2 = sp.optimize.approx_fprime(x0, prior.logpdf, 1e-15)
print(np.linalg.norm(gradeval1-gradeval2)/np.linalg.norm(gradeval2))
# %%
#prior.sample(1)