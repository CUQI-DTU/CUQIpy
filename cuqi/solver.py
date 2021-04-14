import numpy as np
from numpy import linalg as LA
# import matplotlib
# import matplotlib.pyplot as plt
eps = np.finfo(float).eps



#===================================================================
#===================================================================
#===================================================================
class CGLS(object):
    # http://web.stanford.edu/group/SOL/software/cgls/
    # If SHIFT is 0, then CGLS is Hestenes and Stiefel's 
    # conjugate-gradient method for least-squares problems. 
    # If SHIFT is nonzero, the system (A'*A + SHIFT*I)*X = A'*b is solved
    
    def __init__(self, A, b, x0, maxit, tol=1e-6, shift=0):
        self.A = A
        self.b = b
        self.x0 = x0
        self.maxit = maxit
        self.tol = tol        
        self.shift = shift
        if not callable(A):
            self.explicitA = True
        else:
            self.explicitA = False
            
    def solve(self):
        # initial state
        x = self.x0.copy()
        if self.explicitA:
            r = self.b - (self.A @ x)
            s = (self.A.T @ r) - self.shift*x
        else:        
            r = self.b - self.A(x, 1)
            s = self.A(r, 2) - self.shift*x
    
        # initialization
        p = s.copy()
        norms0 = LA.norm(s)
        normx = LA.norm(x)
        gamma, xmax = norms0**2, normx
    
        # main loop
        k, flag, indefinite = 0, 0, 0
        while (k < self.maxit) and (flag == 0):
            k += 1  
            # xold = np.copy(x)
            #
            if self.explicitA:
                q = self.A @ p
            else:
                q = self.A(p, 1)
            delta_cgls = LA.norm(q)**2 + self.shift*LA.norm(p)**2
            #
            if (delta_cgls < 0):
                indefinite = True
            elif (delta_cgls == 0):
                delta_cgls = eps
            alpha_cgls = gamma / delta_cgls
            #
            x += alpha_cgls*p
            x  = np.maximum(x, 0)
            r -= alpha_cgls*q
            if self.explicitA:
                s = self.A.T @ r - self.shift*x     
            else:
                s = self.A(r, 2) - self.shift*x
            #
            gamma1 = gamma.copy()
            norms = LA.norm(s)
            gamma = norms**2
            p = s + (gamma/gamma1)*p
        
            # convergence
            normx = LA.norm(x)
            xmax = max(xmax, normx)
            flag = (norms <= norms0*self.tol) or (normx*self.tol >= 1)
            # resNE = norms / norms0
        #
        shrink = normx/xmax
        # if k == self.maxit:          
        #     flag = 2   # CGLS iterated MAXIT times but did not converge
        #     Warning('\n maxit reached without convergence !')
        if indefinite:          
            flag = 3   # Matrix (A'*A + delta*L) seems to be singular or indefinite
            ValueError('\n Negative curvature detected !')  
        if shrink <= np.sqrt(self.tol):
            flag = 4   # Instability likely: (A'*A + delta*L) indefinite and NORM(X) decreased
            ValueError('\n Instability likely !') 
    
        return x, k



#===================================================================
#===================================================================
#===================================================================
class PDHG(object):
    def __init__(self):
        raise NotImplementedError        

