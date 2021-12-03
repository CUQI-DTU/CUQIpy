import numpy as np
from numpy import linalg as LA
from scipy.optimize import fmin_l_bfgs_b
import scipy.optimize as opt
from cuqi.samples import CUQIarray

class L_BFGS_B(object):
    """Wrapper for :meth:`scipy.optimize.fmin_l_bfgs_b`.

    Minimize a function func using the L-BFGS-B algorithm.
    
    Note, Scipy does not recommend using this method.

    Parameters
    ----------
    func : callable f(x,*args)
        Function to minimize.
    x0 : ndarray
        Initial guess.
    gradfunc : callable f(x,*args), optional
        The gradient of func. 
        If None, then the solver approximates the gradient.
    kwargs : keyword arguments passed to scipy's L-BFGS-B algorithm. See documentation for scipy.optimize.minimize

    Methods
    ----------
    :meth:`solve`: Runs the solver and returns the solution and info about the optimization.
    """
    def __init__(self,func,x0, gradfunc = None, **kwargs):
        self.func= func
        self.x0 = x0
        self.gradfunc = gradfunc
        self.kwargs = kwargs
    
    def solve(self):
        """Runs optimization algorithm and returns solution and info.

        Returns
        ----------
        solution : array_like
            Estimated position of the minimum.
        info : dict
            Information dictionary.
            success: 1 if minimization has converged, 0 if not.
            message: Description of the cause of the termination.
            func: Function value at the estimated minimum.
            grad: Gradient at the estimated minimum.
            nit: Number of iterations.
            nfev: Number of func evaluations.
        """
        # Check if there is a gradient. If not, let the solver use an approximate gradient
        if self.gradfunc is None:
            approx_grad = 1
        else:
            approx_grad = 0
        # run solver
        solution = fmin_l_bfgs_b(self.func,self.x0, fprime = self.gradfunc, approx_grad = approx_grad, **self.kwargs)
        if solution[2]['warnflag'] == 0:
            success = 1
            message = 'Optimization terminated successfully.'
        elif solution[2]['warnflag'] == 1:
            success = 0
            message = 'Terminated due to too many function evaluations or too many iterations.'
        else:
            success = 0
            message = solution[2]['task']
        info = {"success": success,
                "message": message,
                "func": solution[1],
                "grad": solution[2]['grad'],
                "nit": solution[2]['nit'], 
                "nfev": solution[2]['funcalls']}
        return solution[0], info

class minimize(object):
    """Wrapper for :meth:`scipy.optimize.minimize`.

    Minimize a function func using scipy's optimize.minimize module.
    
    Parameters
    ----------
    func : callable f(x,*args)
        Function to minimize.
    x0 : ndarray
        Initial guess.
    gradfunc : callable f(x,*args), optional
        The gradient of func. 
        If None, then the solver approximates the gradient.
    method : str or callable, optional
        Type of solver. Should be one of
        ‘Nelder-Mead’
        ‘Powell’
        ‘CG’
        ‘BFGS’
        ‘Newton-CG’ 
        ‘L-BFGS-B’
        ‘TNC’ 
        ‘COBYLA’ 
        ‘SLSQP’
        ‘trust-constr’
        ‘dogleg’ 
        ‘trust-ncg’ 
        ‘trust-exact’ 
        ‘trust-krylov’ 
        If not given, chosen to be one of BFGS, L-BFGS-B, SLSQP, depending if the problem has constraints or bounds.
    kwargs : keyword arguments passed to scipy's minimizer. See documentation for scipy.optimize.minimize

    Methods
    ----------
    :meth:`solve`: Runs the solver and returns the solution and info about the optimization.
    """
    def __init__(self,func,x0, gradfunc = None, method = None, **kwargs):
        self.func= func
        self.x0 = x0
        self.method = method
        self.gradfunc = gradfunc
        self.kwargs = kwargs
    
    def solve(self):
        """Runs optimization algorithm and returns solution and info.

        Returns
        ----------
        solution : array_like
            Estimated position of the minimum.
        info : dict
            Information dictionary.
            success: 1 if minimization has converged, 0 if not.
            message: Description of the cause of the termination.
            func: Function value at the estimated minimum.
            grad: Gradient at the estimated minimum.
            nit: Number of iterations.
            nfev: Number of func evaluations.
        """
        solution = opt.minimize(self.func, self.x0, jac = self.gradfunc, method = self.method, **self.kwargs)
        info = {"success": solution['success'],
                "message": solution['message'],
                "func": solution['fun'],
                "grad": solution['jac'],
                "nit": solution['nit'], 
                "nfev": solution['nfev']}
        if isinstance(self.x0,CUQIarray):
            sol = CUQIarray(solution['x'],geometry=self.x0.geometry)
        else:
            sol = solution['x']
        return sol, info

class maximize(minimize):
    """Simply calls ::class:: cuqi.solver.minimize with -func."""
    def __init__(self,func,x0, gradfunc = None, method = None, **kwargs):
        def nfunc(*args,**kwargs):
            return -func(*args,**kwargs)
        if gradfunc is not None:
            def ngradfunc(*args,**kwargs):
                return -gradfunc(*args,**kwargs)
        else:
            ngradfunc = gradfunc
        super().__init__(nfunc,x0,ngradfunc,method,**kwargs)

class CGLS(object):
    """Conjugate Gradient method for unsymmetric linear equations and least squares problems.

    See http://web.stanford.edu/group/SOL/software/cgls/ for the matlab version it is based on.

    Solve Ax=b or minimize ||Ax-b||^2 or solve (A^TA+sI)x=A^Tb.
    
    Parameters
    ----------
    A : ndarray or callable f(x,*args).
    b : ndarray.
    x0 : ndarray. Initial guess.
    maxit : The maximum number of iterations.
    tol : The numerical tolerance for convergence checks.
    shift : The shift parameter (s) shown above.

    Methods
    ----------
    :meth:`solve`: Runs the solver and returns the solution.
    """
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
        eps = np.finfo(float).eps

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

            if self.explicitA:
                q = self.A @ p
            else:
                q = self.A(p, 1)
            delta_cgls = LA.norm(q)**2 + self.shift*LA.norm(p)**2

            if (delta_cgls < 0):
                indefinite = True
            elif (delta_cgls == 0):
                delta_cgls = eps
            alpha_cgls = gamma / delta_cgls

            x += alpha_cgls*p
            r -= alpha_cgls*q
            if self.explicitA:
                s = self.A.T @ r - self.shift*x     
            else:
                s = self.A(r, 2) - self.shift*x

            gamma1 = gamma.copy()
            norms = LA.norm(s)
            gamma = norms**2
            p = s + (gamma/gamma1)*p
        
            # convergence
            normx = LA.norm(x)
            xmax = max(xmax, normx)
            flag = (norms <= norms0*self.tol) or (normx*self.tol >= 1)
            # resNE = norms / norms0

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

class PDHG(object):
    """Primal-Dual Hybrid Gradient algorithm."""
    def __init__(self):
        raise NotImplementedError        

