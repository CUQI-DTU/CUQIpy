import numpy as np
from numpy import linalg as LA
from scipy.optimize import fmin_l_bfgs_b, least_squares
import scipy.optimize as opt
import scipy.sparse as spa

from cuqi.array import CUQIarray
from cuqi import config
eps = np.finfo(float).eps

try:
    from sksparse.cholmod import cholesky
    has_cholmod = True
except ImportError:
    has_cholmod = False


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



class LS(object):
    """Wrapper for :meth:`scipy.optimize.least_squares`.

    Solve nonlinear least-squares problems with bounds:
    
    minimize F(x) = 0.5 * sum(rho(f_i(x)**2), i = 0, ..., m-1)
    subject to lb <= x <= ub
    
    Parameters
    ----------
    func : callable f(x,*args)
        Function to minimize.
    x0 : ndarray
        Initial guess.
    Jac : callable f(x,*args), optional
        The Jacobian of func. 
        If None, then the solver approximates the Jacobian.
    loss: callable rho(x,*args)
        Determines the loss function
        'linear' : rho(z) = z. Gives a standard least-squares problem.
        'soft_l1': rho(z) = 2*((1 + z)**0.5 - 1). The smooth approximation of l1 (absolute value) loss.
        'huber'  : rho(z) = z if z <= 1 else 2*z**0.5 - 1. Works similar to 'soft_l1'.
        'cauchy' : rho(z) = ln(1 + z). Severely weakens outliers influence, but may cause difficulties in optimization process.
        'arctan' : rho(z) = arctan(z). Limits a maximum loss on a single residual, has properties similar to 'cauchy'.        
    method : str or callable, optional
        Type of solver. Should be one of
        'trf', Trust Region Reflective algorithm: for large sparse problems with bounds.
        'dogbox', dogleg algorithm with rectangular trust regions, for small problems with bounds.
        'lm', Levenberg-Marquardt algorithm as implemented in MINPACK. Doesn't handle bounds and sparse Jacobians.
    """
    def __init__(self, func, x0, jacfun=None, method='trf', loss='linear', tol=1e-6, maxit=1e4):
        self.func = func
        self.x0 = x0
        self.jacfun = jacfun
        self.method = method
        self.loss = loss
        self.tol = tol
        self.maxit = int(maxit)
    
    def solve(self):
        """Runs optimization algorithm and returns solution and info.

        Returns
        ----------
        solution : Tuple
            Solution found (array_like) and optimization information (dictionary).
        """
        solution = least_squares(self.func, self.x0, jac=self.jacfun, \
                                method=self.method, loss=self.loss, xtol=self.tol, max_nfev=self.maxit)
        info = {"success": solution['success'],
                "message": solution['message'],
                "func": solution['fun'],
                "jac": solution['jac'],
                "nfev": solution['nfev']}
        if isinstance(self.x0, CUQIarray):
            sol = CUQIarray(solution['x'], geometry=self.x0.geometry)
        else:
            sol = solution['x']
        return sol, info



class CGLS(object):
    """Conjugate Gradient method for unsymmetric linear equations and least squares problems.

    See http://web.stanford.edu/group/SOL/software/cgls/ for the matlab version it is based on.
    
    If SHIFT is 0, then CGLS is Hestenes and Stiefel's conjugate-gradient method for least-squares problems. 
    If SHIFT is nonzero, the system (A'*A + SHIFT*I)*X = A'*b is solved.

    Solve Ax=b or minimize ||Ax-b||^2 or solve (A^TA+sI)x=A^Tb.
    
    Parameters
    ----------
    A : ndarray or callable f(x,*args).
    b : ndarray.
    x0 : ndarray. Initial guess.
    maxit : The maximum number of iterations.
    tol : The numerical tolerance for convergence checks.
    shift : The shift parameter (s) shown above.
    """    
    def __init__(self, A, b, x0, maxit, tol=1e-6, shift=0):
        self.A = A
        self.b = b
        self.x0 = x0
        self.maxit = int(maxit)
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


class PCGLS:
    """Conjugate Gradient method for least squares problems with preconditioning
    
    See Bjorck (1996) - Numerical Methods for Least Squares Problems. Pag 294.

    Parameters
    ----------
    A : ndarray or callable f(x,*args).
        Function or array representing the forward model.
    b : ndarray
        Data vector.
    x0 : ndarray
        Initial guess.    
    P : ndarray
        Preconditioner array in sparse format.
    maxit : int
        The maximum number of iterations.
    tol : float
        The numerical tolerance for convergence checks.
    """    
    def __init__(self, A, b, x0, P, maxit, tol=1e-6, shift=0):
        self._A = A
        self._b = b
        self._x0 = x0
        self._P = P
        self._maxit = int(maxit)
        self._tol = tol        
        self._shift = shift
        self._dim = len(x0)
        if not callable(A):
            self._explicitA = True
        else:
            self._explicitA = False
        #
        if self._dim < config.MAX_DIM_INV:
            self._explicitPinv = True
            Pinv = spa.linalg.inv(P)
        else:
            self._explicitPinv = False
            Pinv = None # we do cholesky.solve or sparse.solve with P
            if has_cholmod:
                # turn P into a cholesky object
                P = cholesky(P, ordering_method='natural')
        self._Pinv = Pinv # inverse of the preconditioner as a ndarray or function

    def solve(self):
        # initial state
        x = self._x0.copy()
        r = self._b - self._apply_A(x, 1)
        s = self._apply_Pinv(self._apply_A(r, 2), 2)
        p = s.copy()

        # initial computations        
        norms0 = LA.norm(s)
        normx = LA.norm(x)
        gamma, xmax = norms0**2, normx

        # main loop
        k, flag, indefinite = 0, 0, 0
        while (k < self._maxit) and (flag == 0):
            k += 1
            #
            t = self._apply_Pinv(p, 1)
            q = self._apply_A(t, 1)
            #
            delta_cgls = LA.norm(q)**2
            if (delta_cgls < 0):
                indefinite = True
            elif (delta_cgls == 0):
                delta_cgls = eps
            alpha_cgls = gamma / delta_cgls
            #
            x += alpha_cgls*t
            r -= alpha_cgls*q
            s = self._apply_Pinv(self._apply_A(r, 2), 2)
            #
            norms = LA.norm(s)
            gamma1 = gamma.copy()
            gamma = norms**2
            beta = gamma / gamma1
            p = s + beta*p

            # convergence
            normx = LA.norm(x)
            xmax = max(xmax, normx)
            flag = (norms <= norms0*self._tol) or (normx*self._tol >= 1)
            # resNE = norms / norms0

        shrink = normx/xmax
        if indefinite:          
            flag = 3   # Matrix (A'*A + delta*L) seems to be singular or indefinite
            ValueError('\n Negative curvature detected !')  
        if shrink <= np.sqrt(self._tol):
            flag = 4   # Instability likely: (A'*A + delta*L) indefinite and NORM(X) decreased
            ValueError('\n Instability likely !') 

        return x, k

    def _apply_A(self, x, flag):
        # applies system operator A: forward or adjoint
        if self._explicitA:
            if flag == 1:
                evalu = self._A @ x
            elif flag == 2:
                evalu = self._A.T @ x
        else:
            evalu = self._A(x, flag)
        return evalu

    def _apply_Pinv(self, x, flag):
        # applies the inverse of the preconditioner P: forward or adjoint (see Bjorck (1996) P. 294)
        if self._explicitPinv:
            if flag == 1:
                precond = self._Pinv @ x
            elif flag == 2:
                 precond = self._Pinv.T @ x
        else:
            if has_cholmod:
                if flag == 1:
                    precond = self._P.solve_A(x, use_LDLt_decomposition=False) 
                elif flag == 2:
                    precond = self._P.solve_At(x, use_LDLt_decomposition=False) 
            else:
                if flag == 1:
                    precond = spa.linalg.spsolve(self._P, x) 
                elif flag == 2:
                    precond = spa.linalg.spsolve(self._P.T, x)
        return precond



class LM(object):
    """Levenberg-Marquardt algorithm for nonlinear least-squares problems.
    This is a translation of LevMaq.m from
    https://github.com/bardsleyj/SIAMBookCodes/tree/master/Functions    
    Used in Bardsley (2019) - Computational UQ for inverse problems. SIAM.

    Based Algorithm 3.3.5 from:
    Kelley (1999) - Iterative Methods for Optimization. SIAM.

    Parameters
    ----------
    A : callable f(x,*args).
    x0 : ndarray. Initial guess.
    jacfun : callable Jac(x). Jacobian of func. 
    maxit : The maximum number of iterations.
    tol : The numerical tolerance for convergence checks.
    gradtol : The numerical tolerance for gradient.
    nu0 : default value for nu parameter in the algorithm.
    sparse : True: if 'jacfun' is defined as a sparse matrix, or as a function that 
            supports sparse operations. False: if 'jacfun' is dense.
    """
    def __init__(self, A, x0, jacfun, maxit=1e4, tol=1e-6, gradtol=1e-8, nu0=1e-3, sparse=True):
        self.A = A
        self.x0 = x0
        self.jacfun = jacfun
        self.maxit = int(maxit)
        self.tol = tol
        self.gradtol = gradtol
        self.nu0 = nu0
        self.n = len(x0)
        self.sparse = sparse
        if not callable(A): # also applies for jacfun
            self.explicitA = True
        else:
            self.explicitA = False

    def solve(self):
        x = self.x0
        if self.explicitA:
            r = self.A @ x
            J = self.jacfun @ x
        else:
            r = self.A(x)
            J = self.jacfun(x)
        g = J.T @ r
        ng = LA.norm(g)
        ng0, nu = np.copy(ng), np.copy(ng)
        f = 0.5*(r.T @ r)
        i = 0

        # solve function depending on sparsity
        if self.sparse:
            insolve = lambda A, b: spa.linalg.spsolve(A, b)
            I = spa.identity(self.n)
        else:
            insolve = lambda A, b: LA.solve(A, b)
            I = np.identity(self.n)

        # parameters required for the LM parameter update
        mu0, mulow, muhigh = 0, 0.25, 0.75
        omdown, omup = 0.5, 2
        while ((ng/ng0) > self.gradtol) and (i < self.maxit):
            i += 1
            s = insolve((J.T@J + nu*I), g)
            xtemp = x-s
            if self.explicitA:
                rtemp = self.A @ xtemp
                Jtemp = self.jacfun @ xtemp
            else:
                rtemp = self.A(xtemp)
                Jtemp = self.jacfun(xtemp)
            ftemp = 0.5*(rtemp.T @ rtemp)

            # LM parameter update
            num, den = f-ftemp, (xtemp-x).T @ g
            if (num != 0) and (den != 0): 
                ratio = -2*(num / den)
            else:
                ratio = 0
            if (ratio < mu0):
                nu = max(omup*nu, self.nu0)
            else:
                x, r, f = np.copy(xtemp), np.copy(rtemp), np.copy(ftemp)
                if self.sparse:
                    J = spa.csr_matrix.copy(Jtemp)
                else:
                    J = np.copy(Jtemp)
                if (ratio < mulow):
                    nu = max(omup*nu, self.nu0)
                elif (ratio > muhigh):
                    nu = omdown*nu
                    if (nu < self.nu0):
                        nu = 0
            g = J.T @ r
            ng = LA.norm(g)

        info = {"func": r,
                "Jac": J,
                "nfev": i}
        return x, info



class PDHG(object):
    """Primal-Dual Hybrid Gradient algorithm."""
    def __init__(self):
        raise NotImplementedError



class FISTA(object):
    """Fast Iterative Shrinkage-Thresholding Algorithm for regularized least squares problems.
    
    Reference:
    Beck, Amir, and Marc Teboulle. "A fast iterative shrinkage-thresholding algorithm for linear inverse problems." SIAM journal on imaging sciences 2.1 (2009): 183-202.

    Minimize ||Ax-b||^2 + f(x).
    
    
    Parameters
    ----------
    A : ndarray or callable f(x,*args).
    b : ndarray.
    x0 : ndarray. Initial guess.
    proximal : callable f(x, gamma) for proximal mapping.
    maxit : The maximum number of iterations.
    stepsize : The stepsize of the gradient step.
    abstol : The numerical tolerance for convergence checks.
    adapative : Whether to use FISTA or ISTA.

    Example
    -----------
    .. code-block:: python
    
        from cuqi.solver import FISTA,  ProximalL1
        import scipy as sp
        import numpy as np

        rng = np.random.default_rng()

        m, n = 10, 5
        A = rng.standard_normal((m, n))
        b = rng.standard_normal(m)
        stepsize = 0.99/(sp.linalg.interpolative.estimate_spectral_norm(A)**2)
        x0 = np.zeros(n)
        fista = FISTA(A, b, x0, proximal = ProximalL1, stepsize = stepsize, maxit = 100, abstol=1e-12, adaptive = True)
        sol, _ = fista.solve()

    """  
    def __init__(self, A, b, x0, proximal, maxit=100, stepsize=1e0, abstol=1e-14, adaptive = True):
        
        self.A = A
        self.b = b
        self.x0 = x0
        self.proximal = proximal
        self.maxit = int(maxit)
        self.stepsize = stepsize
        self.abstol = abstol
        self.adaptive = adaptive

    @property
    def _explicitA(self):
        return not callable(self.A)
            
    def solve(self):
        # initial state
        x = self.x0.copy()
        stepsize = self.stepsize
        
        k = 0
        
        while True:
            x_old = x.copy()
            k += 1
        
            if self._explicitA:
                grad = self.A.T@(self.A @ x_old - self.b)
            else:
                grad = self.A(self.A(x_old, 1) - self.b, 2)
                
            x_new = self.proximal(x_old-stepsize*grad, stepsize)
                        
            if LA.norm(x_new-x_old) <= self.abstol or (k >= self.maxit):
                return x_new, k
            
            if self.adaptive:
                x_new = x_new + ((k-1)/(k+2))*(x_new - x_old)
              
            x = x_new.copy()
    
    
def ProjectNonnegative(x):
    """(Euclidean) projection onto the nonnegative orthant.
    
    Parameters
    ----------
    x : array_like.
    """  
    return np.maximum(x, 0)

def ProjectBox(x, lower = None, upper = None):
    """(Euclidean) projection onto a box.
    
    Parameters
    ----------
    x : array_like.
    lower : array_like. Lower bound of box. Zero if None.
    upper : array_like. Upper bound of box. One if None.
    """  
    if lower is None:
        lower = np.zeros_like(x)
    
    if upper is None:
        upper = np.ones_like(x)
    
    return np.minimum(np.maximum(x, lower), upper)

def ProximalL1(x, gamma):
    """(Euclidean) proximal operator of the \|x\|_1 norm.
    Also known as the shrinkage or soft thresholding operator.
    
    Parameters
    ----------
    x : array_like.
    gamma : scale parameter.
    """
    return np.multiply(np.sign(x), np.maximum(np.abs(x)-gamma, 0))
