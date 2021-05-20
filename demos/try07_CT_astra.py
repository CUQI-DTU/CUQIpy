# =================================================================
# Created by:
# Felipe Uribe @ DTU
# =================================================================
# Version 2021
# =================================================================
import sys
sys.path.append("../")
import time
import numpy as np
import scipy as sp
import scipy.stats as sps
import scipy.special as spe
import scipy.io as spio
import matplotlib
import matplotlib.pyplot as plt
#
import astra
from data.projection_functions import A
from data.CGLS_temp import CGLS_reg_fix, CGLS_reg_samples

# =================================================================
# params
# =================================================================
N = 150           # object size N-by-N pixels
p = int(1.5*N)    # number of detector pixels
q = 90            # number of projection angles
theta = np.linspace(0, 2*np.pi, q, endpoint=False)   # view angles

# =================================================================
# prior for the attenuation coefficient (x)
# =================================================================
d = N*N                 # dimension
mu_pr_x = np.zeros(d)   # prior mean
I = sp.sparse.identity(N, format='csc')

# 1D finite difference matrix with Neumann BCs
D = sp.sparse.spdiags([-np.ones(N), np.ones(N)], [0, 1], N, N).tocsr()
D[-1,-1] = 0

# 2D finite differences in each direction
D1 = sp.sparse.kron(I, D)
D2 = sp.sparse.kron(D, I)

# prior: Laplace 
dbar = d-1
# def logpi_pr_x(delta, x): 
#     return d*(np.log(delta)-np.log(2)) - delta*(np.linalg.norm(D1@x, ord=1)+np.linalg.norm(D2@x, ord=1))

# we approximate it as a Gaussian with structure
vareps = 1e-6
varepsvec = vareps*np.ones(d)
def Lk_fun(x_k): # returns L_1 and L_2 structure matrices
    diag1, diag2 = 1/np.sqrt((D1 @ x_k)**2 + varepsvec), 1/np.sqrt((D2 @ x_k)**2 + varepsvec)
    W1, W2 = sp.sparse.diags(diag1), sp.sparse.diags(diag2)    
    return ((D1.T @ (W1 @ D1)) + (D2.T @ (W2 @ D2))), (W1.sqrt() @ D1), (W2.sqrt() @ D2)

# =================================================================
# hyperprior precision: delta
# =================================================================
# 1-dimensional gamma distribution
alpha_2, beta_2 = 1, 1e-4
# const2 = alpha_2*np.log(beta_2) - spe.loggamma(alpha_2)
# def logpi_pr_d(delta): 
#     return const2 + (alpha_2-1)*np.log(delta) - beta_2*delta

# =================================================================
# load data and set-up likelihood
# =================================================================
# 'b_meas','b_meas_i','e_true','std_e','x_true'
data = spio.loadmat('data/data_true_grains_150.mat')

# underlying true
x_true = data['x_true']
x_truef = x_true.flatten(order='F')
xt_norm = np.linalg.norm(x_truef)

# noise and data
m = q*p                          # number of data points
b_data = data['b_meas'].flatten()
sigma_noi = data['std_e'][0]
lambd_obs = 1/(sigma_noi**2)

# Gaussian likelihood
m_log_2pi = m*np.log(2*np.pi)   # constant in the MVN
def logpi_like(x):
    misfit = b_data - A(x, 1)
    loglike_eval = -0.5*(-m*(np.log(lambd_obs)) + lambd_obs*(misfit.T @ misfit) + m_log_2pi)
    return loglike_eval

# =================================================================
# conditional inverse scale X: delta (sample)
# =================================================================
# conditional precision of X: delta (sample)
def pi_cond_delta_rnd(b, N): 
    return np.random.gamma(shape=dbar+alpha_2, scale=1/(b+beta_2), size=(N))

# =================================================================
# Hybrid Gibbs
# =================================================================
# for CGLS
x_tol, n_cgls = 1e-4, 10

# samples
n_s = int(2e3)          # number of samples in Gibbs sampler
n_b = int(0.2*n_s)      # burn-in
n_t = n_s+n_b           # total number of samples

# allocation
x_s = np.zeros((d, n_t))
delta_s = np.zeros(n_t)

# initial states
_, W1sq_D1, W2sq_D2 = Lk_fun(mu_pr_x)
x0, nit0 = CGLS_reg_fix(mu_pr_x, W1sq_D1, W2sq_D2, b_data, lambd_obs, 1, n_cgls, x_tol)
#
L, W1sq_D1, W2sq_D2 = Lk_fun(x0)
x_s[:, 0] = x0
delta_s[0] = pi_cond_delta_rnd(x0.T @ (L @ x0), 1)

# relative errors
e_x = np.zeros(n_t)
e_x[0] = np.linalg.norm(x_s[:, 0]-x_truef)/xt_norm

# ================================================================
np.random.seed(1)
print('\n***MCMC***\n')

st = time.time()
for s in range(n_t-1):
    print('*sample: ', s+1, '/', n_t, '\t')
    
    # ===X========================================================
    x_s[:, s+1], _ = CGLS_reg_samples(x_s[:, s], W1sq_D1, W2sq_D2, b_data, \
                                lambd_obs, delta_s[s], n_cgls, x_tol)
    e_x[s+1] = np.linalg.norm(x_s[:, s+1]-x_truef)/xt_norm

    # ===update Laplace approx====================================
    L, W1sq_D1, W2sq_D2 = Lk_fun(x_s[:, s+1])

    # ===hyperparam===============================================
    # inverse scale
    delta_s[s+1] = pi_cond_delta_rnd(x_s[:, s+1].T @ (L @ x_s[:, s+1]), 1)
    
    # astra.clear() # free memory after reconstruction and projection

print('\nElapsed time:', time.time()-st, '\n')   
# mdict = {'X':x_s, 'err_x':e_x, 'delta_s':delta_s}
# spio.savemat('grainsx1_UQ_ns10.mat', mdict)

# main stats
x_mean = np.mean(x_s[:, n_b:], axis=1)
x_std = np.std(x_s[:, n_b:], axis=1)
err = np.linalg.norm(x_mean-x_truef)/xt_norm
print('\n relerr:', err)

# =================================================================
# plots
# =================================================================
fig = plt.figure(figsize=(10,4))
ax1 = plt.subplot(131)
cs = ax1.imshow(x_true.reshape(N,N), extent=[0, 1, 0, 1], aspect='equal', cmap='YlGnBu_r')
cbar = fig.colorbar(cs, ax=ax1, shrink=0.5)
ax1.set_title('True')
ax1.tick_params(axis='both', which='both', length=0)
plt.setp(ax1.get_xticklabels(), visible=False)
plt.setp(ax1.get_yticklabels(), visible=False)

ax2 = plt.subplot(132)
cs = ax2.imshow(x_mean.reshape(N,N).T, extent=[0, 1, 0, 1], aspect='equal', cmap='YlGnBu_r')
cbar = fig.colorbar(cs, ax=ax2, shrink=0.5)
# ax.set_title('True')
ax2.tick_params(axis='both', which='both', length=0)
plt.setp(ax2.get_xticklabels(), visible=False)
plt.setp(ax2.get_yticklabels(), visible=False)

ax2 = plt.subplot(133)
cs = ax2.imshow(x_std.reshape(N,N).T, extent=[0, 1, 0, 1], aspect='equal', cmap='YlGnBu_r')
cbar = fig.colorbar(cs, ax=ax2, shrink=0.5)
# ax.set_title('True')
ax2.tick_params(axis='both', which='both', length=0)
plt.setp(ax2.get_xticklabels(), visible=False)
plt.setp(ax2.get_yticklabels(), visible=False)

plt.tight_layout()
plt.show()