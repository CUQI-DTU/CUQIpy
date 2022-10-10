# =================================================================
# Created by:
# Felipe Uribe @ DTU
# =================================================================
# Version 2022
# =================================================================
import sys
import numpy as np
import time
sys.path.append("..") 
import numpy as np
#
import cuqi
import matplotlib.pyplot as plt

# =================================================================
tp = cuqi.testproblem.WangCubic()
n = tp.model.domain_dim
forward = tp.model.forward
y_data = tp.data

# transformation
mu_pr = tp.prior.mean
def T(u):
    x = mu_pr + u
    return x

# Jacobian of the transformation
In = np.identity(n)
def J_fun(u):
    return In

# =================================================================
# sample
# ================================================================
np.random.seed(1)
Ns = int(3e3)
Nb = int(0.1*Ns)
u0 = np.random.randn(n)

# set sampler
smp = 'NUTS'
if smp == 'RTO':
    MCMC = cuqi.sampler._rto.RTO(tp.posterior, T, J_fun, mode='MAP', unadjusted=False, x0=u0)
elif smp == 'NUTS':
    MCMC = cuqi.sampler.NUTS(tp.posterior, u0, max_depth=12)

# sample
tic = time.time()
sol = MCMC.sample(Ns, Nb)
toc = time.time()-tic
print('\nElapsed time\n:', toc) 

if smp == 'RTO':
    uchain = sol.samples.T
    uchain_prop = sol.samples_prop.T
    # transform back to the original space
    x_chain = T(uchain)
    x_chain_prop = T(uchain_prop)
elif smp == 'NUTS':
    x_chain = sol.samples.T

mu = np.mean(x_chain, axis=0)
print(mu) 

# ================================================================
fig, axes = plt.subplots(2, 1, figsize=(12, 4))
ax1, ax2 = axes.flatten()
ax1.plot(x_chain[:, 0], 'k-', linewidth=1)
ax1.set_xlim(0, Ns)
ax1.set_ylabel(r'$x_1$')
#
ax2.plot(x_chain[:, 1], 'k-', linewidth=1)
ax2.set_xlim(0, Ns)
ax2.set_ylabel(r'$x_2$')
#
if smp == 'RTO':
    ax1.plot(x_chain_prop[:, 0], 'r--', linewidth=1, alpha=0.5)
    ax2.plot(x_chain_prop[:, 1], 'r--', linewidth=1, alpha=0.5)
plt.tight_layout()

#################
n1 = 100
xx = np.linspace(-2, 2, n1)
X, Y = np.meshgrid(xx, xx)
grid = np.vstack([X.flatten(), Y.flatten()]).T
n = grid.shape[0]
#
pi_pos_eval, pi_pr_eval = np.empty(n), np.empty(n)
for i in range(n):
    pi_pr_eval[i] = tp.prior.pdf(grid[i, :])
    pi_pos_eval[i] = tp.posterior.pdf(grid[i, :])

fig = plt.figure()
cs1 = plt.contour(X, Y, pi_pr_eval.reshape(n1, n1), 15, cmap='Blues')
cs2 = plt.contour(X, Y, pi_pos_eval.reshape(n1, n1), 20, cmap='Reds')
plt.plot(x_chain[:, 0], x_chain[:, 1], 'k.', markersize=1.5, label='accepted')
if smp == 'RTO':
    plt.plot(x_chain_prop[:, 0], x_chain_prop[:, 1], 'ro', fillstyle='none', markersize=3, label='proposed')
plt.legend()
plt.xlim(-2, 2)
plt.ylim(-1, 2)
fig.gca().set_aspect("equal")
plt.tight_layout()
# plt.savefig('sol_mean.pdf', format='pdf', dpi=150, bbox_inches='tight')
plt.show()