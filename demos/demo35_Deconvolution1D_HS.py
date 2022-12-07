#%% This file show-cases the Deconvolution 2D test problem.
import sys
sys.path.append("..")
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import cuqi

# ===================================================================
# deconvolution problem
# ===================================================================
TP = cuqi.testproblem.Deconvolution1D(noise_type='normalizedgaussian', \
                        kernel_param=55, phantom='pc', noise_std=0.03)
d = TP.model.domain_dim
grid = TP.model.domain_geometry.grid
f_true = TP.exactSolution
Aop = lambda x: TP.model.forward(x)

# ===================================================================
# data (and likelihood)
y_true = TP.exactData
y_data = TP.data
m = len(y_data)
# likelihood = TP.likelihood
# sigma_obs = np.sqrt(likelihood.distribution.cov)
# lambd_obs = 1/(sigma_obs**2)

# ===================================================================
mu_pr_x = np.zeros(d)   # prior mean

# 1D finite difference matrix with zero BCs
one_vec = np.ones(d)
D = sp.sparse.diags([-one_vec, one_vec], offsets=[-1, 0], shape=(d+1, d), format='csc', dtype=int)
D = D[:-1, :]

# for x: least-squares form
def proj_forward_reg(x, flag, tau, sigma, lambd):
    # regularized ASTRA projector [A; Lsq @ W] w.r.t. x.np.add.outer(X,Y)
    Wsq = sp.sparse.diags(1/(tau*sigma))
    if flag == 1:
        out1 = np.sqrt(lambd) * Aop(x, 1) # A @ x
        out2 = (Wsq @ D) @ x
        out = np.hstack([out1, out2])
    else:
        out1 = np.sqrt(lambd) * Aop(x[:m], 2) # A.T @ b
        out2 = (Wsq @ D).T @ x[m:]
        out = out1 + out2
    return out


# ===================================================================
plt.figure()
plt.plot(grid, f_true, 'b-')
plt.plot(grid, y_data, 'r*')
plt.plot(grid, y_true, 'k-')
plt.show()