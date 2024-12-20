import numpy as np
import scipy as sp

from cuqi.solver import ScipyLBFGSB, ScipyMinimizer, ScipyLeastSquares, CGLS, LM, FISTA, ADMM, ProximalL1, ProjectNonnegative
from scipy.optimize import lsq_linear


def test_CGLS():
    # Parameters
    rng = np.random.default_rng()
    m, n = 1000, 500
    A = sp.sparse.rand(m, n, density=1e-4, random_state=rng)
    b = rng.standard_normal(m)
    res = lsq_linear(A, b, tol=1e-8)
    ref_sol = res.x
    #
    x0 = np.zeros(n)
    sol, _ = CGLS(A, b, x0, int(1e4), tol=1e-8).solve()

    # Compare
    assert np.allclose(sol, ref_sol, rtol=1e-3)

def test_ScipyMinimizer_without_gradient():
    solver = ScipyMinimizer(sp.optimize.rosen,
                            np.array([1.3, 0.7, 0.8, 1.9, 1.2]),
                            method='Nelder-Mead',
                            tol=1e-6)
    sol, _ = solver.solve()
    sol_ref = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    assert np.allclose(sol, sol_ref)

def test_ScipyMinimizer_with_gradient():
    solver = ScipyMinimizer(sp.optimize.rosen,
                            np.array([1.3, 0.7, 0.8, 1.9, 1.2]),
                            gradfunc=sp.optimize.rosen_der,
                            method='BFGS',
                            tol=1e-6)
    sol, _ = solver.solve()
    sol_ref = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    assert np.allclose(sol, sol_ref)

def test_ScipyLBFGSB_without_gradient():
    solver = ScipyLBFGSB(sp.optimize.rosen,
                         np.array([1.3, 0.7, 0.8, 1.9, 1.2]))
    sol, _ = solver.solve()
    sol_ref = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    assert np.allclose(sol, sol_ref)

def test_ScipyLBFGSB_with_gradient():
    solver = ScipyLBFGSB(sp.optimize.rosen,
                         np.array([1.3, 0.7, 0.8, 1.9, 1.2]),
                         gradfunc=sp.optimize.rosen_der)
    sol, _ = solver.solve()
    sol_ref = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    assert np.allclose(sol, sol_ref)

def test_ScipyLeastSquares_without_Jac():
    def fun_rosenbrock(x):
        return np.array([10 * (x[1] - x[0]**2), (1 - x[0])])
    x0 = np.array([2, 2])
    solver = ScipyLeastSquares(fun_rosenbrock, x0)
    sol, _ = solver.solve()
    sol_ref = np.array([1, 1])
    assert np.allclose(sol, sol_ref)

def test_ScipyLeastSquares_with_Jac():
    def fun_rosenbrock(x):
        return np.array([10 * (x[1] - x[0]**2), (1 - x[0])])
    def jac_rosenbrock(x):
        return np.array([
            [-20 * x[0], 10],
            [-1, 0]])
    x0 = np.array([2, 2])
    solver = ScipyLeastSquares(fun_rosenbrock, x0, jacfun=jac_rosenbrock)
    sol, _ = solver.solve()
    sol_ref = np.array([1, 1])
    assert np.allclose(sol, sol_ref)

def test_LM():
    # compare to MATLAB's original code solution
    t = np.arange(1, 10, 2)
    A = lambda x: x[0]*(1-np.exp(-x[1]*t))
    x_true = np.array([1, 0.1]) #  true parameter values
    N = len(x_true)
    #
    b = np.array([0.0886724443121281, 0.270993439737938, 0.38588480731453, 0.492318565823575, 0.584974827859323])
    M = len(b)
    #    
    lam = 1
    x0 = np.ones(N)
    Q = np.eye(M)
    residual = lambda x: Q.T @ (np.sqrt(lam)*(A(x)-b))
    jacobian = lambda x: Q.T @ (np.sqrt(lam)*np.array([1-np.exp(-x[1]*t), x[0]*t*np.exp(-x[1]*t)]).T)
    opt, info = LM(residual, x0, jacobian, maxit=int(1e4), tol=1e-3, gradtol=1e-8, sparse=False).solve()
    MSE = np.linalg.norm(info['func'])**2/(M-N)

    # solution TwoParameters.m from John's book
    ref_MSE = 8.06826104975311e-05
    ref_opt = np.array([0.92953293909007, 0.109186388510998])

    assert np.allclose(MSE, ref_MSE) and np.allclose(opt, ref_opt)
    
def test_FISTA():
    # Parameters
    rng = np.random.default_rng(seed = 42)
    m, n = 10, 5
    A = rng.standard_normal((m, n))
    b = rng.standard_normal(m)
    
    stepsize = 0.99/(sp.linalg.interpolative.estimate_spectral_norm(A)**2)
    x0 = np.zeros(n)
    sol, _ = FISTA(A, b, ProximalL1, x0, stepsize = stepsize, maxit = 100, abstol=1e-12, adaptive = True).solve()

    ref_sol = np.array([-1.83273787e-03, -1.72094582e-13,  0.0, -3.35835639e-01, -1.27795593e-01])
    # Compare
    assert np.allclose(sol, ref_sol, atol=1e-4)

def test_ADMM_matrix_form():
    # Parameters
    rng = np.random.default_rng(seed = 42)
    m, n = 10, 5
    A = rng.standard_normal((m, n))
    b = rng.standard_normal(m)
    
    k = 4
    L = rng.standard_normal((k, n))

    x0 = np.zeros(n)
    sol, _ = ADMM(A, b, [(ProximalL1, np.eye(n)), (lambda z, _ : ProjectNonnegative(z), L)],
                   x0, 10, maxit = 100, adaptive = True).solve()

    ref_sol = np.array([-3.99513417e-03, -1.32339656e-01, -4.52822633e-02, -7.44973888e-02, -3.35005208e-11])
    # Compare
    assert np.allclose(sol, ref_sol, atol=1e-4)


def test_ADMM_function_form():
    # Parameters
    rng = np.random.default_rng(seed = 42)
    m, n = 10, 5
    A = rng.standard_normal((m, n))
    def A_fun(x, flag):
        if flag == 1:
            return A@x
        if flag == 2:
            return A.T@x
        
    b = rng.standard_normal(m)
    
    k = 4
    L = rng.standard_normal((k, n))

    x0 = np.zeros(n)
    sol, _ = ADMM(A_fun, b, [(ProximalL1, np.eye(n)), (lambda z, _ : ProjectNonnegative(z), L)],
                   x0, 10, maxit = 100, adaptive = True).solve()

    print(sol)
    ref_sol = np.array([-3.99513417e-03, -1.32339656e-01, -4.52822633e-02, -7.44973888e-02, -3.35005208e-11])
    # Compare
    assert np.allclose(sol, ref_sol, atol=1e-4)
