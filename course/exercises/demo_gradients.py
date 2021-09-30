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
import scipy
import matplotlib.pyplot as plt

# myfuns
import cuqi

d = 10
mean_pr = np.zeros(d)
sigma_pr = 5
corrmat_pr = np.eye(d)
prior = cuqi.distribution.GaussianGen(mean_pr, (sigma_pr, corrmat_pr))

x0 = np.random.multivariate_normal(size=d)
eval1 = prior.logpdf(x0)
eval2 = scipy.stats.multivariate_normal(mean_pr, (sigma_pr**2)*corrmat_pr)
