# =================================================================
# Created by:
# Felipe Uribe @ DTU
# =================================================================
import time
import sys
sys.path.append("..") 
import numpy as np
import scipy as sp
import scipy.stats as sps
#
import cuqi
import matplotlib.pyplot as plt

# ===================================================================
# deconvolution problem
# ===================================================================
TP = cuqi.testproblem.deblur()
n = TP.model.domain_dim
x_true = TP.exactSolution
data = TP.data
xx = TP.mesh
        
# Extract Likelihood
likelihood = TP.likelihood

# Gaussian i.i.d. prior
prior = cuqi.distribution.GaussianCov(mean=np.zeros(n), cov=1)

# posterior
posterior = cuqi.distribution.Posterior(likelihood, prior)

# ===================================================================
# define transformation to standard Gaussian and its Jacobian
# ===================================================================
# the original prior is Laplace diff
delta = 10  # prior scale

# 1D finite difference matrix with zero BCs
D = sp.sparse.diags([-np.ones(n), np.ones(n)], offsets=[-1, 0], shape=(n+1, n), format='csc', dtype=int)
D = D[:-1, :] # D[0,-1] = 1
Dinv = sp.sparse.tril(np.ones((n, n), dtype=int)) #sp.sparse.linalg.inv(D) # 

# transform and its inverse
def T(u):
    Dx = sps.laplace.ppf(sps.norm.cdf(u), scale=1/delta)
    # Dx = np.array([sps.laplace.ppf(sps.norm.cdf(u[i]), scale=1/delta) for i in range(n)])
    # Dx = np.array([-(1/delta)*np.sign(u[i])*np.log(1-2*np.abs(sps.norm.cdf(u[i])-0.5)) for i in range(n)])
    x = Dinv @ Dx
    return x

# Jacobian of the transformation
def J_fun(u):
    diag = sps.norm.pdf(u) / (delta*sps.norm.cdf(-np.abs(u)))
    # diag = np.array([(sps.norm.pdf(u[i]))/(delta*sps.norm.cdf(-np.abs(u[i]))) for i in range(n)])
    Jac0 = sp.sparse.spdiags(diag, 0, n, n).tocsc()
    Jac = Dinv @ Jac0
    return Jac

# ================================================================
# RTO
# ================================================================
np.random.seed(1)
Ns = int(1e3)
Nb = int(0.1*Ns)
u0 = np.random.randn(n)

# set sampler
MCMC = cuqi.sampler.RTO(posterior, T, J_fun, mode='MAP', unadjusted=True, x0=u0)
# MCMC = cuqi.sampler.NUTS(tp.posterior, u0, max_depth=12)

# sample
tic = time.time()
sol = MCMC.sample(Ns, Nb)
uchain = sol.samples
uchain_prop = sol.samples_prop
toc = time.time()-tic
print('\nElapsed time\n:', toc) 

# transform back to the original space
x_chain = np.array([T(uchain[:, i]) for i in range(Ns)])
x_chain_prop = np.array([T(uchain_prop[:, i]) for i in range(Ns)])
x_mean = np.mean(x_chain, axis=0)
x_std = np.std(x_chain, axis=0, ddof=1)

# ================================================================
plt.figure()
plt.plot(xx, x_true, 'b-')
plt.plot(xx, x_mean, 'r-')
# plt.fill_between(xx, CI_95_x[0, :], CI_95_x[1, :], color='r', alpha=0.3)
plt.plot(xx, x_mean-x_std, 'r--')
plt.plot(xx, x_mean+x_std, 'r--')
plt.tight_layout()
plt.show()