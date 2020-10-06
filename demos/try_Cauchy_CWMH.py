# =============================================================================
# Created by:
# Felipe Uribe @ DTU
# =============================================================================
# Version 2020-10
# =============================================================================
import time
import numpy as np
import scipy as sp
import scipy.sparse
import matplotlib
import matplotlib.pyplot as plt

# myfuns
from deblur import deblur
from MCMC_algorithms import CWMH

# =============================================================================
# set-up the discrete convolution model
# =============================================================================
n = 128
bnds = np.array([0, 1])
t_mid = np.linspace(bnds[0], bnds[1], n)
h = t_mid[1]-t_mid[0]

# set-up computational model kernel
a = 48
kernel = lambda x, y, a: a / 2*np.exp(-a*abs((x-y)))   # blurring kernel

# convolution matrix
T1, T2 = np.meshgrid(t_mid, t_mid)
A = h*kernel(T1, T2, a)
maxval = A.max()
A[A < 5e-3*maxval] = 0
A = sp.sparse.csc_matrix(A)   # make A sparse

# forward model
forward = lambda x: A @ x

# =============================================================================
# data and noise
# =============================================================================
# compute truth and noisy convolved data
sigma_obs = 0.1
b, f_true, g_true = deblur(t_mid, kernel, sigma_obs, n)
norm_f = np.linalg.norm(f_true)

# Gaussian likelihood params 
m = len(b)                             # number of data points 
m_log_2pi = m*np.log(2*np.pi)          # constant in the MVN
lambd_obs = 1/(sigma_obs**2)           # precision parameter
log_det_obs = m*(-np.log(lambd_obs))   # np.log( (1/lambd_obs)**m )
def likelihood_logpdf(x):
    misfit = forward(x) - b
    loglike_eval = -0.5*( log_det_obs + lambd_obs*(misfit.T @ misfit) + m_log_2pi )
    # = sp.stats.multivariate_normal.logpdf(g_data, (A @ x), (1/lambd_obs)*np.eye(dim))
    if np.isinf(loglike_eval):
        raise RuntimeError('ERROR: INF loglike')  
    return loglike_eval

# =============================================================================
# prior
# =============================================================================
# 1D finite difference matrix
D = sp.sparse.spdiags([-np.ones(n), np.ones(n)], [0, 1], n-1, n)

# Cauchy difference prior
delta = 0.4
gamma = delta*h      # scale parameter: "regularization"
x0 = np.zeros(n)     # mode
def prior_logpdf(x):
    Dx = D @ (x-x0)
    log_pi_eval = -len(Dx)*np.log(np.pi) + sum( np.log(gamma) - np.log(Dx**2 + gamma**2) )
    # check sp.stats.cauchy.logpdf(Dx, 0, delta)
    return log_pi_eval

# =============================================================================
# Gaussian prior
# sigma_pr    = 0.315
# lambd_pr    = 1/(sigma_pr**2)
# dim_log_2pi = n*np.log(2*np.pi)
# log_det     = n*(-np.log(lambd_pr))   # np.log( (1/lambd)**dim )
# def prior_logpdf(x):
#     log_pi_eval = -0.5*( log_det + lambd_pr*(x.T @ x) + dim_log_2pi )
#     # = sp.stats.multivariate_normal.logpdf(x, np.zeros(dim), (1/lambd)*np.eye(dim))
#     return log_pi_eval

# Tikhonov: stacked
# Gamma = sigma_pr*np.eye(n)
# A_big = np.vstack([A, Gamma])
# b_big = np.hstack([b, np.zeros(n)])
# MAP_pos = np.linalg.lstsq(A_big, b_big, rcond=None)[0]# rank = np.linalg.matrix_rank(A)
# #
# relerr = round(np.linalg.norm(MAP_pos - f_true)/norm_f*100)
# print('relerror MAP:',relerr)

# =============================================================================
# posterior sampling
# =============================================================================
target = lambda x: likelihood_logpdf(x) + prior_logpdf(x)
#
Ns = int(1e4)
# proposal = lambda N, x_t, sigma: np.random.multivariate_normal(x_t, (sigma**2)*np.eye(n), N).flatten()
proposal = lambda N, x_t, sigma: np.random.normal(x_t, sigma, size=(N,n))
scale = 0.03*np.ones(n)
#
ti = time.time()
x_s, target_eval, acc = CWMH(Ns, target, proposal, x0, scale)
print('\nElapsed time:', time.time() - ti)

# =============================================================================
mu_xpos = x_s.mean(axis=1)
med_xpos = np.median(x_s, axis=1) # sp.stats.mode
sigma_xpos = x_s.std(axis=1)
lo95 = np.percentile(x_s, 2.5, axis=1)
up95 = np.percentile(x_s, 97.5, axis=1)
#
relerr1 = round(np.linalg.norm(mu_xpos - f_true)/norm_f*100, 2)
relerr2 = round(np.linalg.norm(med_xpos - f_true)/norm_f*100, 2)
print('\nrelerror mean:', relerr1, '\nrelerror median:', relerr2)

# =============================================================================
# plots
# =============================================================================
plt.figure()
plt.plot(t_mid, f_true, 'k-')
plt.plot(t_mid, g_true, 'b-')
plt.plot(t_mid, b, 'r.')
plt.tight_layout()

plt.figure()
plt.plot(target_eval)
plt.xlabel('Sample index')
plt.ylabel('Target function values')
plt.tight_layout()

plt.figure()
plt.plot(t_mid, f_true,'-', color='forestgreen', linewidth=3, label='True')
plt.plot(t_mid, mu_xpos,'-', color='navy', label='mean')
plt.plot(t_mid, med_xpos,'--', color='crimson', label='median')
plt.fill_between(t_mid, up95, lo95, color='dodgerblue',alpha=0.25)
plt.legend(loc = 'upper right', shadow = False, ncol = 1, fancybox = True, prop = {'size':15})
plt.xticks(np.linspace(bnds[0], bnds[1],5))
plt.xlim([bnds[0], bnds[1]])
plt.ylim(-0.5,3.5)
plt.tight_layout()
# plt.savefig('fig.png', format='png', dpi=150, bbox_inches='tight')
plt.show()