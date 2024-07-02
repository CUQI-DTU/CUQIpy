# =================================================================
# Created by:
# Felipe Uribe @ DTU
# =================================================================
# Version 2022
# =================================================================
import sys
sys.path.append("..") 
import numpy as np
import scipy as sp
import scipy.stats as sps
#
import cuqi
import matplotlib.pyplot as plt

# =================================================================
# sample a 100D Gaussian
d = 100
mu_q = np.zeros(d)
sigma_q = np.linspace(0.01, 1, d)
Lambd_q = sp.sparse.spdiags(1/(sigma_q**2), 0, d, d)

# ===================================================================
# unormalized target
def logpi_target(x):
    return 0.5*( - (x.T @ Lambd_q @ x) )
def grad_logpi_target(x):
    return -(Lambd_q @ x)

target = cuqi.distribution.UserDefinedDistribution(d, logpi_target, grad_logpi_target)

# =================================================================
# sample
np.random.seed(1)
x0 = mu_q #np.random.rand(d)
Ns = int(2e3)
Nb = int(5e2)

# HMC
traject_length = 150
MCMC = cuqi.sampler.HMC(target, x0, adapt_traject_length=traject_length)
solution_hmc = MCMC.sample(Ns, Nb)
x_chain_hmc = solution_hmc.samples.T
steps_hmc = solution_hmc.acc_rate
mu_hmc = np.mean(x_chain_hmc, axis=0)
sigma_hmc = np.std(x_chain_hmc, axis=0, ddof=1)

# NUTS
MCMC = cuqi.sampler.NUTS(target, x0)
solution_nuts = MCMC.sample(Ns, Nb)
x_chain_nuts = solution_nuts.samples.T
steps_nuts = solution_nuts.acc_rate
mu_nuts = np.mean(x_chain_nuts, axis=0)
sigma_nuts = np.std(x_chain_nuts, axis=0, ddof=1)

# =================================================================
# plots
# ================================================================
plt.figure()
plt.plot(steps_nuts, 'bo', markersize=1.5, label='NUTS')
plt.plot(steps_hmc, 'r.', markersize=1.5, label='HMC')
plt.title('step sizes')
plt.xlabel('iterations')
plt.ylabel('epsilon')
plt.legend()
plt.tight_layout()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
ax1.plot(x_chain_nuts[:, -1], 'b.', label='NUTS')
ax1.set_title('NUTS chain')
ax1.set_xlabel('iterations')
ax1.set_ylabel('theta_100')
ax1.legend()
ax2.plot(x_chain_hmc[:, -1], 'r.', label='HMC')
ax2.set_title('HMC chain')
ax2.set_xlabel('iterations')
ax2.set_ylabel('theta_100')
ax2.legend()
plt.tight_layout()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.plot(sigma_q, mu_nuts, 'bo', label='NUTS')
ax1.plot(sigma_q, mu_hmc, 'r.', label='HMC')
ax1.legend()
ax1.set_ylabel('sample mean at each coordinate')
ax1.set_xlim(0, 1)
ax1.set_ylim(-0.6, 0.6)
ax2.plot(sigma_q, sigma_nuts, 'bo', label='NUTS')
ax2.plot(sigma_q, sigma_hmc, 'r.', label='HMC')
ax2.set_ylabel('sample std at each coordinate')
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1.1)
fig.suptitle('NUTS')
fig.suptitle('stats comparison')
plt.tight_layout()
plt.show()
