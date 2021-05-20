# ========================================================================
# Created by:
# Felipe Uribe @ DTU compute
# ========================================================================
# Version 2020-06
# ========================================================================
import sys
import numpy as np
from numpy import linalg as LA
import scipy as sp
#
from data.projection_functions import proj_forward_sino, proj_backward_sino
eps = np.finfo(float).eps

# from numba.typed import List
# from numba import njit

# =========================================================================
# =========================================================================
# =========================================================================
def CGLS_reg_fix(x_old, W1sq_D1, W2sq_D2, b_meas, lambd, delta, x_maxit, x_tol):  
    # params for cgls
    m = len(b_meas)
    nbar = len(x_old)

    # apply cgls
    G_fun = lambda x, flag: proj_forward_reg(x, flag, m, W1sq_D1, W2sq_D2, lambd, delta)
    g = np.hstack([np.sqrt(lambd)*b_meas, np.zeros(nbar), np.zeros(nbar)])
    x_next, it = cgls(G_fun, g, x_old, x_maxit, x_tol)

    return x_next, it


# =========================================================================
# =========================================================================
# =========================================================================
def CGLS_reg_samples(x_old, W1sq_D1, W2sq_D2, b_meas, lambd, delta, x_maxit, x_tol):  
    # params for cgls
    m = len(b_meas)
    nbar = len(x_old)

    # apply cgls
    G_fun = lambda x, flag: proj_forward_reg(x, flag, m, W1sq_D1, W2sq_D2, lambd, delta)
    g = np.hstack([np.sqrt(lambd)*b_meas, np.zeros(nbar), np.zeros(nbar)]) + np.random.randn(m+2*nbar)
    x_next, it = cgls(G_fun, g, x_old, x_maxit, x_tol)
    
    return x_next, it


# =========================================================================
# =========================================================================
# =========================================================================
def proj_forward_reg(x, flag, m, W1sq_D1, W2sq_D2, lambd, delta):
    # regularized ASTRA projector [A; Lsq] w.r.t. x.
    if flag == 1:
        out1 = np.sqrt(lambd) * proj_forward_sino(x) # A @ x
        out2 = np.sqrt(delta) * (W1sq_D1 @ x)
        out3 = np.sqrt(delta) * (W2sq_D2 @ x)
        out  = np.hstack([out1, out2, out3])
    else:
        idx = int(len(x[m:])/2)
        out1 = np.sqrt(lambd) * proj_backward_sino(x[:m]) # A.T @ b
        out2 = np.sqrt(delta) * (W1sq_D1.T @ x[m:m+idx])
        out3 = np.sqrt(delta) * (W2sq_D2.T @ x[m+idx:])
        out  = out1 + out2 + out3
    return out


#=========================================================================
#=========================================================================
#=========================================================================
# @njit
def cgls(A, b, x0, maxit, tol):
    # http://web.stanford.edu/group/SOL/software/cgls/
    
    # initial state
    x = x0.copy()
    r = b - A(x, 1)
    s = A(r, 2) #- shift*x
    
    # initialization
    p = s.copy()
    norms0 = LA.norm(s)
    normx = LA.norm(x)
    gamma, xmax = norms0**2, normx
    
    # main loop
    k, flag, indefinite = 0, 0, 0
    while (k < maxit) and (flag == 0):
        k += 1  
        # xold = np.copy(x)
        #
        q = A(p, 1)
        delta_cgls = LA.norm(q)**2 #+ shift*LA.norm(p)**2
        #
        if (delta_cgls < 0):
            indefinite = 1
        elif (delta_cgls == 0):
            delta_cgls = eps
        alpha_cgls = gamma / delta_cgls
        #
        x += alpha_cgls*p    
        x  = np.maximum(x, 0)
        r -= alpha_cgls*q
        s  = A(r, 2) #- shift*x
        #
        gamma1 = gamma.copy()
        norms = LA.norm(s)
        gamma = norms**2
        p = s + (gamma/gamma1)*p
        
        # convergence
        normx = LA.norm(x)
        # relerr = LA.norm(x - xold) / normx
        # if relerr <= tol:
        #     flag = 1
        xmax = max(xmax, normx)
        flag = (norms <= norms0*tol) or (normx*tol >= 1)
        # flag = 1: CGLS converged to the desired tolerance TOL within MAXIT
        # resNE = norms / norms0
    #
    shrink = normx/xmax
    if k == maxit:          
        flag = 2   # CGLS iterated MAXIT times but did not converge
    if indefinite:          
        flag = 3   # Matrix (A'*A + delta*L) seems to be singular or indefinite
        sys.exit('\n Negative curvature detected !')  
    if shrink <= np.sqrt(tol):
        flag = 4   # Instability likely: (A'*A + delta*L) indefinite and NORM(X) decreased
        sys.exit('\n Instability likely !')  
    
    return x, k