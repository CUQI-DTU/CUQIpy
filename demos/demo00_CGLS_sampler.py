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
from cuqi.sampler import CGLS_sampler

# =============================================================================
# set-up the discrete convolution model
# =============================================================================
test = cuqi.testproblem.Deblur()
n = test.model.dim[1]
tt = test.t
h = test.meshsize

# =============================================================================
# data and noise
# =============================================================================
# compute truth and noisy convolved data
norm_f = np.linalg.norm(test.f_true)

# Gaussian likelihood params
b = test.data
m = len(b)       # number of data points
lambd = 1/(test.noise.std**2)

# model matrix
A = test.model.get_matrix()

# =============================================================================
# prior
# =============================================================================
mu_pr = np.zeros(n)
delta = 25             # prior precision parameter
prior = cuqi.distribution.GMRF(mu_pr, delta, n, 1, 'zero')

# =============================================================================
# posterior sampling
# =============================================================================
# run sampler
Ns = int(5e3)      # number of samples
Nb = 0             # burn-in
x0 = mu_pr
maxit = 20
#
MCMC = CGLS_sampler(b, test.model, test.noise, prior, x0, maxit)
#
ti = time.time()
x_s = MCMC.sample(Ns, Nb)
print('Elapsed time:', time.time() - ti)

# =============================================================================
# sampler solution
mean_xpos = x_s.mean(axis=1) # sp.stats.mode
sigma_xpos = x_s.std(axis=1)
lo95, up95 = mean_xpos+sigma_xpos, mean_xpos-sigma_xpos#np.percentile(x_s, [2.5, 97.5], axis=1)
relerr = round(np.linalg.norm(mean_xpos - test.f_true)/norm_f*100, 2)
print('\nRelerror median:', relerr, '\n')

# analytical solution (uncomment when A is sparse)
# Lambd_ex = lambd*(A.T @ A) + delta*prior.L
# Sigma_ex = sp.sparse.linalg.inv(Lambd_ex)
# sigma_ex = np.sqrt(Sigma_ex.diagonal())
# mean_ex = sp.sparse.linalg.spsolve(Lambd_ex, lambd*(A.T @ b))

# =============================================================================
# plots
# =============================================================================
plt.figure()
plt.plot(tt, test.f_true, '-', color='forestgreen', linewidth=3, label='True')
plt.plot(tt, mean_xpos, '--', color='blue', label='mean samp')
plt.fill_between(tt, up95, lo95, color='dodgerblue', alpha=0.25)

# uncomment when A is sparse
# plt.plot(tt, mean_ex, '-', color='crimson', label='mean ex')
# plt.plot(tt, mean_ex+sigma_ex, '--', color='crimson')
# plt.plot(tt, mean_ex-sigma_ex, '--', color='crimson')

plt.legend(loc='upper right', shadow=False, ncol = 1, fancybox=True, prop={'size':15})
plt.xticks(np.linspace(tt[0], tt[-1], 5))
plt.xlim([tt[0], tt[-1]])
plt.ylim(-0.5, 3.5)
plt.tight_layout()
# plt.savefig('fig.png', format='png', dpi=150, bbox_inches='tight')
plt.show()
