# %%
# =============================================================================
# Created by:
# Felipe Uribe @ DTU
# =============================================================================
# Version 2020-10
# =============================================================================
import sys
sys.path.append("../")
import time
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# myfuns
import cuqi
from cuqi.sampler import LinearRTO

# =============================================================================
# set-up the discrete convolution model
# =============================================================================
test = cuqi.testproblem._Deblur()
n = test.model.domain_dim
tt = test.mesh
h = test.meshsize

# =============================================================================
# data and noise
# =============================================================================
# compute truth and noisy convolved data
norm_f = np.linalg.norm(test.exactSolution)

# =============================================================================
# prior
# =============================================================================
mu_pr = np.zeros(n)
delta = 25             # prior precision parameter
prior = cuqi.distribution.GMRF(mu_pr, delta, 'zero')

# =============================================================================
# posterior sampling
# =============================================================================
# run sampler
Ns = int(5e3)      # number of samples
Nb = 0             # burn-in
x0 = mu_pr
maxit = 20
#
posterior = cuqi.distribution.Posterior(test.likelihood, prior)
MCMC = LinearRTO(posterior, x0, maxit)
#
ti = time.time()
x_s = MCMC.sample(Ns, Nb).samples
print('Elapsed time:', time.time() - ti)

# =============================================================================
# sampler solution
mean_xpos = x_s.mean(axis=1) # sp.stats.mode
sigma_xpos = x_s.std(axis=1)
lo95, up95 = mean_xpos+sigma_xpos, mean_xpos-sigma_xpos#np.percentile(x_s, [2.5, 97.5], axis=1)
relerr = round(np.linalg.norm(mean_xpos - test.exactSolution)/norm_f*100, 2)
print('\nRelerror median:', relerr, '\n')

# =============================================================================
# plots
# =============================================================================
plt.figure()
plt.plot(tt, test.exactSolution, '-', color='forestgreen', linewidth=3, label='True')
plt.plot(tt, mean_xpos, '--', color='blue', label='mean samp')
plt.fill_between(tt, up95, lo95, color='dodgerblue', alpha=0.25)
plt.legend(loc='upper right', shadow=False, ncol = 1, fancybox=True, prop={'size':15})
plt.xticks(np.linspace(tt[0], tt[-1], 5))
plt.xlim([tt[0], tt[-1]])
plt.ylim(-0.5, 3.5)
plt.tight_layout()
# plt.savefig('fig.png', format='png', dpi=150, bbox_inches='tight')
plt.show()
