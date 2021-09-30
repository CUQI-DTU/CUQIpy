# %%
# =============================================================================
# Created by:
# Felipe Uribe @ DTU
# =============================================================================
# Version 2020-10
# =============================================================================
import sys
sys.path.append("../../")
import time
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.sparse import linalg as splinalg
from scipy.linalg import eigh

# myfuns
import cuqi
%load_ext autoreload
%autoreload 2

# %%
d = 3000
mean_pr = np.zeros(d)
sigma_pr = 5
corrmat_pr = np.eye(d)
prior = cuqi.distribution.GaussianGen(mean_pr, sigma_pr**2*corrmat_pr)

# %%
x0 = 1000*np.random.rand(d)
eval1 = prior.logpdf(x0)
eval2 = sp.stats.multivariate_normal.logpdf(x0, mean_pr, (sigma_pr**2)*corrmat_pr)

# %%
gradeval1 = prior.gradient(x0)
gradeval2 = sp.optimize.approx_fprime(x0, prior.logpdf, 1e-15)

# %%
