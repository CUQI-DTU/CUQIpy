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
logpi_like = tp.likelihood.log
pi_pos = tp.posterior.pdf

# =================================================================
# plots
# ================================================================
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
plt.xlim(-2, 2)
plt.ylim(-1, 2)
fig.gca().set_aspect("equal")
plt.tight_layout()
plt.show()

# ADD TEST: likelihood gradient checks
#
# m = 1 # number of data points
# y_data = 1 # measured data
# sigma_obs = 1
# prec_obs = 1
# def logpi_like(x, grad=False):
#     forward_eval = forward(x)
#     misfit = y_data - forward_eval
#     if grad:
#         grad_eval = gradient(x)
#         return -0.5*(prec_obs * misfit**2 + np.log(2*np.pi) + np.log(sigma_obs)), prec_obs * misfit * grad_eval
#     else:
#         return -0.5*(prec_obs * misfit**2 + np.log(2*np.pi) + np.log(sigma_obs))#sps.norm.logpdf(y_data, forward_eval, sigma_obs)
#
# x0 = np.random.randn(2)
# eval1, gradeval1 = logpi_like(x0, True)
# #
# eval2 = loglike(x0)
# gradeval2 = tp.likelihood.gradient(x0)
# #
# loglike = lambda x: logpi_like(x)
# gradevalu3 = sp.optimize.approx_fprime(x0, loglike, 1e-5)