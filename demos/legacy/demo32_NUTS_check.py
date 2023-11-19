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
tp = cuqi.testproblem.WangCubic()
forward = tp.model.forward
y_data = tp.data

# Bayes
pi_pr = tp.prior.pdf
pi_pos = tp.posterior.pdf

# =================================================================
# sample
# ================================================================
np.random.seed(1)
x0 = np.ones(tp.model.domain_dim)
Ns = int(1e4)
Nb = int(0.5*Ns)
MCMC = cuqi.sampler.NUTS(tp.posterior, x0, max_depth = 12)
solution = MCMC.sample(Ns, Nb)
x_chain = solution.samples.T
steps = solution.acc_rate
mu = np.mean(x_chain, axis=0)
print(mu)

# =================================================================
# plots
# ================================================================
plt.figure()
plt.plot(steps, 'b.', markersize=1.5)
plt.xlim(0, Nb+1000)

fig, axes = plt.subplots(2, 1, figsize=(12, 4))
ax1, ax2 = axes.flatten()
ax1.plot(x_chain[:, 0], 'k-')
ax1.set_xlim(0, Ns)
# ax1.set_ylim(-0.1, 1.6)
ax1.set_ylabel(r'$x_1$')
#
ax2.plot(x_chain[:, 1], 'k-')
ax2.set_xlim(0, Ns)
# ax1.set_ylim(-0.1, 1.6)
ax2.set_ylabel(r'$x_2$')
plt.tight_layout()

#################
n1 = 100
xx = np.linspace(-2, 2, n1)
X, Y = np.meshgrid(xx, xx)
grid = np.vstack([X.flatten(), Y.flatten()]).T
n = grid.shape[0]
#
pi_pr_eval = np.empty(n)
pi_pos_eval = np.empty(n)
for i in range(n):
    pi_pr_eval[i] = pi_pr(grid[i, :])
    pi_pos_eval[i] = pi_pos(grid[i, :])
#
fig = plt.figure()
cs1 = plt.contour(X, Y, pi_pr_eval.reshape(n1,n1), 15, cmap='Blues')
cs2 = plt.contour(X, Y, pi_pos_eval.reshape(n1,n1), 20, cmap='Reds')
plt.plot(x_chain[:, 0], x_chain[:, 1], 'k.', markersize=1)
plt.xlim(-2, 2)
plt.ylim(-1, 2)
fig.gca().set_aspect("equal")
plt.tight_layout()
plt.show()