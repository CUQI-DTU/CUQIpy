import scipy as sp
import numpy as np
import cuqi
from cuqi.solver import CGLS, LM, LS
from cuqi.sampler import Sampler

# try importing fast sparse libraries
try:
    import sparseqr 
    spaqr = True
except ImportError:
    spaqr = False
try:
    from sksparse.cholmod import cholesky
    chols = True
except ImportError:
    chols = False

class Linear_RTO(Sampler):
    """
    Linear RTO (Randomize-Then-Optimize) sampler.

    Samples posterior related to the inverse problem with Gaussian likelihood and prior, and where the forward model is Linear.

    Parameters
    ------------
    target : `cuqi.distribution.Posterior` or 5-dimensional tuple.
        If target is of type cuqi.distribution.Posterior, it represents the posterior distribution.
        If target is a 5-dimensional tuple, it assumes the following structure:
        (data, model, L_sqrtprec, P_mean, P_sqrtrec)
        
        Here:
        data: is a m-dimensional numpy array containing the measured data.
        model: is a m by n dimensional matrix or LinearModel representing the forward model.
        L_sqrtprec: is the squareroot of the precision matrix of the Gaussian likelihood.
        P_mean: is the prior mean.
        P_sqrtprec: is the squareroot of the precision matrix of the Gaussian mean.

    x0 : `np.ndarray` 
        Initial point for the sampler. *Optional*.

    maxit : int
        Maximum number of iterations of the inner CGLS solver. *Optional*.

    tol : float
        Tolerance of the inner CGLS solver. *Optional*.

    callback : callable, *Optional*
        If set this function will be called after every sample.
        The signature of the callback function is `callback(sample, sample_index)`,
        where `sample` is the current sample and `sample_index` is the index of the sample.
        An example is shown in demos/demo31_callback.py.
        
    """
    def __init__(self, target, x0=None, maxit=10, tol=1e-6, shift=0, **kwargs):
        
        # Accept tuple of inputs and construct posterior
        if isinstance(target, tuple) and len(target) == 5:
            # Structure (data, model, L_sqrtprec, P_mean, P_sqrtprec)
            data = target[0]
            model = target[1]
            L_sqrtprec = target[2]
            P_mean = target[3]
            P_sqrtprec = target[4]

            # If numpy matrix convert to CUQI model
            if isinstance(model, np.ndarray) and len(model.shape) == 2:
                model = cuqi.model.LinearModel(model)

            # Check model input
            if not isinstance(model, cuqi.model.LinearModel):
                raise TypeError("Model needs to be cuqi.model.LinearModel or matrix")

            # Likelihood
            L = cuqi.distribution.GaussianSqrtPrec(model, L_sqrtprec).to_likelihood(data)

            # Prior TODO: allow multiple priors stacked
            #if isinstance(P_mean, list) and isinstance(P_sqrtprec, list):
            #    P = cuqi.distribution.JointGaussianSqrtPrec(P_mean, P_sqrtprec)
            #else:
            P = cuqi.distribution.GaussianSqrtPrec(P_mean, P_sqrtprec)

            # Construct posterior
            target = cuqi.distribution.Posterior(L, P)

        super().__init__(target, x0=x0, **kwargs)

        # Check target type
        if not isinstance(target, cuqi.distribution.Posterior):
            raise ValueError(f"To initialize an object of type {self.__class__}, 'target' need to be of type 'cuqi.distribution.Posterior'.")       

        # Check Linear model and Gaussian prior+likelihood
        if not isinstance(self.model, cuqi.model.LinearModel):
            raise TypeError("Model needs to be linear")

        if not hasattr(self.likelihood.distribution, "sqrtprec"):
            raise TypeError("Distribution in Likelihood must contain a sqrtprec attribute")

        if not hasattr(self.prior, "sqrtprec"):
            raise TypeError("prior must contain a sqrtprec attribute")

        if not hasattr(self.prior, "sqrtprecTimesMean"):
            raise TypeError("Prior must contain a sqrtprecTimesMean attribute")

        # Modify initial guess        
        if x0 is not None:
            self.x0 = x0
        else:
            self.x0 = np.zeros(self.prior.dim)

        # Other parameters
        self.maxit = maxit
        self.tol = tol        
        self.shift = 0
                
        L1 = self.likelihood.distribution.sqrtprec
        L2 = self.prior.sqrtprec
        L2mu = self.prior.sqrtprecTimesMean

        # pre-computations
        self.m = len(self.data)
        self.n = len(self.x0)
        self.b_tild = np.hstack([L1@self.data, L2mu]) 

        if not callable(self.model):
            self.M = sp.sparse.vstack([L1@self.model, L2])
        else:
            # in this case, model is a function doing forward and backward operations
            def M(x, flag):
                if flag == 1:
                    out1 = L1 @ self.model.forward(x)
                    out2 = L2 @ x
                    out  = np.hstack([out1, out2])
                elif flag == 2:
                    idx = int(self.m)
                    out1 = self.model.adjoint(L1.T@x[:idx])
                    out2 = L2.T @ x[idx:]
                    out  = out1 + out2                
                return out   
            self.M = M       

    @property
    def prior(self):
        return self.target.prior

    @property
    def likelihood(self):
        return self.target.likelihood

    @property
    def model(self):
        return self.target.model     
    
    @property
    def data(self):
        return self.target.data

    def _sample(self, N, Nb):   
        Ns = N+Nb   # number of simulations        
        samples = np.empty((self.n, Ns))
                     
        # initial state   
        samples[:, 0] = self.x0
        for s in range(Ns-1):
            y = self.b_tild + np.random.randn(len(self.b_tild))
            sim = CGLS(self.M, y, samples[:, s], self.maxit, self.tol, self.shift)            
            samples[:, s+1], _ = sim.solve()

            self._print_progress(s+2,Ns) #s+2 is the sample number, s+1 is index assuming x0 is the first sample
            self._call_callback(samples[:, s+1], s+1)

        # remove burn-in
        samples = samples[:, Nb:]
        
        return samples, None, None

    def _sample_adapt(self, N, Nb):
        return self._sample(N,Nb)

#====================================================================
#====================================================================
#====================================================================
class RTO(Sampler):
    """
    RTO (Randomize-Then-Optimize) sampler.

    Samples posterior related to an inverse problem with Gaussian likelihood.
    Other priors can be used after a suitable transformation to the standard Gaussian space.
    Model can be nonlinear.

    Parameters
    ------------
    target : `cuqi.distribution.Posterior` or 3-dimensional tuple.
        If target is of type cuqi.distribution.Posterior, it represents the posterior distribution.
        If target is a 3-dimensional tuple, it assumes the following structure:
        (data, model, L_sqrtprec)
        
        Here:
        data: is a m-dimensional numpy array containing the measured data.
        model: is a m by n dimensional matrix or LinearModel representing the forward model.
        L_sqrtprec: is the squareroot of the precision matrix of the Gaussian likelihood.

    x0 : `np.ndarray` 
        Initial point for the sampler. *Optional*.

    maxit : int
        Maximum number of iterations of the inner solver. *Optional*.

    tol : float
        Tolerance of the inner solver. *Optional*.

    callback : callable, *Optional*
        If set this function will be called after every sample.
        The signature of the callback function is `callback(sample, sample_index)`,
        where `sample` is the current sample and `sample_index` is the index of the sample.
        An example is shown in demos/demo31_callback.py.
        
    """
    def __init__(self, target, T, J_fun, mode='MAP', unadjusted=False, x0=None, maxit=100, tol=1e-4, **kwargs):

        # Accept tuple of inputs and construct posterior
        if isinstance(target, tuple) and len(target) == 3:
            # Structure (data, model, L_sqrtprec, P_mean, P_sqrtprec)
            data = target[0]
            model = target[1]
            L_sqrtprec = target[2]

            # If numpy matrix convert to CUQI model
            if isinstance(model, np.ndarray) and len(model.shape) == 2:
                model = cuqi.model.LinearModel(model)

            # Likelihood
            L = cuqi.distribution.GaussianSqrtPrec(model, L_sqrtprec).to_likelihood(data)

            # Prior: standard Gaussian
            P = cuqi.distribution.GaussianCov(mean=np.zeros(self.n), cov=1)

            # Construct posterior
            target = cuqi.distribution.Posterior(L, P)

        super().__init__(target, x0=x0, **kwargs)

        # Check target type
        if not isinstance(target, cuqi.distribution.Posterior):
            raise ValueError(f"To initialize an object of type {self.__class__}, 'target' need to be of type 'cuqi.distribution.Posterior'.")       

        if not hasattr(self.likelihood.distribution, "sqrtprec"):
            raise TypeError("Distribution in Likelihood must contain a sqrtprec attribute")

        # Modify initial guess        
        if x0 is not None:
            self.x0 = x0
        else:
            self.x0 = np.zeros(self.prior.dim)

        # Other parameters
        self.maxit = maxit
        self.tol = tol
        self.T = T
        self.J_fun = J_fun
        if hasattr(self.data, '__len__'):
            self.m = len(self.data)
        else:
            self.m = 1
        if hasattr(self.x0, '__len__'):
            self.n = len(self.x0)
        else:
            self.n = 1
        #
        In = sp.sparse.eye(self.n, dtype=int)
        L1 = self.likelihood.distribution.sqrtprec

        if not callable(self.model):
            # in this case, model is a function is linear
            def M(u, e):
                x = self.T(u)   # nonlinear transformation
                r = np.hstack([L1*((self.model@x) - self.data), u]) - e   # residual
                return r
            def Jac(u):
                JT = self.J_fun(u)
                J = sp.sparse.vstack([L1*self.model@JT, In])
                return J
        else:
            # in this case, model is a function
            def M(u, e):
                x = self.T(u)   # nonlinear transformation
                r = np.hstack([L1*(self.model(x) - self.data), u]) - e   # residual
                return r
            def Jac(u):
                dT_du = self.J_fun(u)
                dF_dT = self.model.gradient(self.T(u), dT_du)
                J = sp.sparse.vstack([L1*dF_dT, In])
                return J

            # store
            self.M = M
            self.Jac = Jac

        # find the MAP and matrix Q for RTO proposal
        self.e0 = np.zeros(self.m+self.n)
        u0 = np.ones(self.n)
        if mode.lower() == 'map':
            print('\nQ from the Jacobian at the MAP...')
            utrg, _, _ = self._solve(lambda u: M(u, self.e0), u0, Jac, tol=1e-6, maxit=int(3e4), method='LS')
            Jeval = Jac(utrg)
        elif mode.lower() == 'prior':
            print('\nQ equal identity...')
            Jeval = sp.sparse.eye(self.m+self.n, dtype=int)
        elif mode.lower() == 'prerun':
            print('\nQ from the Jacobian at the mean of "prerunned" unadjusted samples...')
            N = int(3e2)
            uMAP, _, _ = self._solve(lambda u: M(u, self.e0), u0, Jac, tol=1e-6, maxit=int(3e4), method='LS')
            Jeval = Jac(uMAP)
            self.Q, _ = self._Qfun(Jeval)
            self.unadjusted = True
            usamp = self.sample(N, 0)
            utrg = np.mean(usamp.samples, axis=1)
            Jeval = Jac(utrg)
            print('Done !\n')
        else:
            raise TypeError("mode has to be: 'MAP', 'prior', or 'prerun'")
        self.Q, _ = self._Qfun(Jeval)
        self.unadjusted = unadjusted
        # plt.figure(1)
        # plt.plot(self.T(uMAP), 'b--')
        # plt.plot(self.T(utrg), 'r--')
        # plt.show()

    @property
    def prior(self):
        return self.target.prior

    @property
    def likelihood(self):
        return self.target.likelihood

    @property
    def model(self):
        return self.target.model     

    @property
    def data(self):
        return self.target.data

    def _sample(self, N, Nb):
        Ns = N+Nb   # number of simulations

        # allocation
        samples = np.empty((self.dim, Ns))
        samples_prop = np.empty((self.dim, Ns))
        logq_RTO_eval = np.empty(Ns)
        acc = np.zeros(Ns, dtype=int)
        ite = np.zeros(Ns, dtype=int)

        # initial state    
        samples[:, 0] = self.x0
        logq_RTO_eval[0], _, _ = self._RTO_proposal(samples[:, 0], 'eval')
        acc[0] = 1

        # run MCMC
        for s in range(Ns-1):
            if self.unadjusted:
                samples[:, s+1], acc[s+1], ite[s+1] = self._single_update_unadjusted(\
                    samples[:, s])
            else:
                samples[:, s+1], samples_prop[:, s+1], logq_RTO_eval[s+1], acc[s+1], ite[s+1] = self._single_update(\
                samples[:, s], logq_RTO_eval[s])
            self._print_progress(s+2, Ns) #s+2 is the sample number, s+1 is index assuming x0 is the first sample
            self._call_callback(samples[:, s+1], s+1)

        # remove burn-in
        samples = samples[:, Nb:]
        samples_prop = samples_prop[:, Nb:]
        logq_RTO_eval = logq_RTO_eval[Nb:]
        accave = acc[Nb:].mean()   
        print('\nAverage acceptance rate:', accave, '\n')
        return samples, logq_RTO_eval, accave, ite, samples_prop

    def _single_update(self, theta_t, logq_RTO_t):
        # propose state
        theta_star, nres, it = self._RTO_proposal(theta_t, 'sample')   # sample from the proposal
        logq_RTO_star, _, _ = self._RTO_proposal(theta_star, 'eval')

        # ratio and acceptance probability
        log_prop_ratio = logq_RTO_t - logq_RTO_star
        log_alpha = min(0, log_prop_ratio)

        # accept/reject
        log_u = np.log(np.random.rand())
        if (log_u < log_alpha) and (nres < 1e-8):
            x_next = theta_star
            logq_RTO_eval_next = logq_RTO_star
            acc = 1
        else:
            x_next = theta_t
            logq_RTO_eval_next = logq_RTO_t
            acc = 0
        # plt.figure(2)
        # plt.plot(self.T(x_next))
        # plt.pause(1)
        return x_next, theta_star, logq_RTO_eval_next, acc, it
    
    def _single_update_unadjusted(self, theta_t):
        # propose state
        theta_star, _, it = self._RTO_proposal(theta_t, 'sample')   # sample from the proposal
        # logq_RTO_star, _, _ = self._RTO_proposal(theta_star, 'eval')

        return theta_star, 1, it

    def _RTO_proposal(self, u_k, flag):
        if (flag == 'eval'):
            r = self.M(u_k, self.e0)
            QtJ = self.Q.T @ self.Jac(u_k)
            logdet = self._logdet(QtJ)
            #
            val = logdet + 0.5*(np.linalg.norm(r)**2 - np.linalg.norm(self.Q.T @ r)**2) # Eq. 6.30 John's
            nres, it = [], []
        elif (flag == 'sample'):
            e = np.random.randn(self.m+self.n)
            fun_k = lambda u: self.Q.T @ self.M(u, e)
            Jac_k = lambda u: self.Q.T @ self.Jac(u)
            val, nres, it = self._solve(fun_k, u_k, Jac_k, self.tol, self.maxit)            
        return val, nres, it

    def _solve(self, fun, u, Jac, tol, maxit, method='LS'):
        if method == 'LS':
            opt, info = LS(fun, u, Jac, tol=tol, maxit=maxit).solve()
        elif method == 'LM':
            opt, info = LM(fun, u, Jac, tol=tol, maxit=maxit).solve()
        val, Qtr, it = opt, info['func'], info['nfev']
        nres = np.linalg.norm(Qtr)**2
        return val, nres, it
    
    def _Qfun(self, J):
        if spaqr:
            Q, R, _, _ = sparseqr.qr(J, economy=True)
        else:
            Q, R = np.linalg.qr(J.todense(), mode='reduced')
            Q = sp.sparse.csc_matrix(Q)
        return Q, R

    def _logdet(self, QtJ):
        if spaqr: 
            _, R, = self._Qfun(QtJ)     
            logdet = np.sum(np.log(np.abs(R.diagonal())))
        elif chols:
            chol = cholesky(QtJ.T @ QtJ, ordering_method='natural')
            logdet = 0.5*chol.logdet()
        else:
            logdet = np.sum(np.log(np.diag(np.linalg.cholesky((QtJ.T @ QtJ).todense()))))
        return logdet

    def _sample_adapt(self, N, Nb):
        return self._sample(N, Nb)