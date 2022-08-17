import scipy as sp
import numpy as np
import cuqi
from cuqi.solver import CGLS
from cuqi.sampler import Sampler


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
