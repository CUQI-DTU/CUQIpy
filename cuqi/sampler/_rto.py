import scipy as sp
from scipy.linalg.interpolative import estimate_spectral_norm
from scipy.sparse.linalg import LinearOperator as scipyLinearOperator
import numpy as np
import cuqi
from cuqi.solver import CGLS, FISTA
from cuqi.sampler import Sampler


class LinearRTO(Sampler):
    """
    Linear RTO (Randomize-Then-Optimize) sampler.

    Samples posterior related to the inverse problem with Gaussian likelihood and prior, and where the forward model is linear or more generally affine.

    Parameters
    ------------
    target : `cuqi.distribution.Posterior`, `cuqi.distribution.MultipleLikelihoodPosterior` or 5-dimensional tuple.
        If target is of type cuqi.distribution.Posterior or cuqi.distribution.MultipleLikelihoodPosterior, it represents the posterior distribution.
        If target is a 5-dimensional tuple, it assumes the following structure:
        (data, model, L_sqrtprec, P_mean, P_sqrtrec)
        
        Here:
        data: is a m-dimensional numpy array containing the measured data.
        model: is a m by n dimensional matrix, AffineModel or LinearModel representing the forward model.
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
            if not isinstance(model, cuqi.model.AffineModel):
                raise TypeError("Model needs to be cuqi.model.AffineModel or matrix")

            # Likelihood
            L = cuqi.distribution.Gaussian(model, sqrtprec=L_sqrtprec).to_likelihood(data)

            # Prior TODO: allow multiple priors stacked
            #if isinstance(P_mean, list) and isinstance(P_sqrtprec, list):
            #    P = cuqi.distribution.JointGaussianSqrtPrec(P_mean, P_sqrtprec)
            #else:
            P = cuqi.distribution.Gaussian(P_mean, sqrtprec=P_sqrtprec)

            # Construct posterior
            target = cuqi.distribution.Posterior(L, P)

        super().__init__(target, x0=x0, **kwargs)

        self._check_posterior()

        # Modify initial guess        
        if x0 is not None:
            self.x0 = x0
        else:
            self.x0 = np.zeros(self.prior.dim)

        # Other parameters
        self.maxit = maxit
        self.tol = tol        
        self.shift = 0
                
        L1 = [likelihood.distribution.sqrtprec for likelihood in self.likelihoods]
        L2 = self.prior.sqrtprec
        L2mu = self.prior.sqrtprecTimesMean

        # pre-computations
        self.n = len(self.x0)
        self.b_tild = np.hstack([L@(likelihood.data - model._shift) for (L, likelihood, model) in zip(L1, self.likelihoods, self.models)]+ [L2mu])

        callability = [callable(likelihood.model) for likelihood in self.likelihoods]
        notcallability = [not c for c in callability]
        if all(notcallability):
            self.M = sp.sparse.vstack([L@likelihood.model for (L, likelihood) in zip(L1, self.likelihoods)] + [L2])
        elif all(callability):
            # in this case, model is a function doing forward and backward operations
            def M(x, flag):
                if flag == 1:
                    out1 = [L @ likelihood.model._forward_func_no_shift(x) for (L, likelihood) in zip(L1, self.likelihoods)] # Use forward function which excludes shift
                    out2 = L2 @ x
                    out  = np.hstack(out1 + [out2])
                elif flag == 2:
                    idx_start = 0
                    idx_end = 0
                    out1 = np.zeros(self.n)
                    for likelihood in self.likelihoods:
                        idx_end += len(likelihood.data)
                        out1 += likelihood.model._adjoint_func_no_shift(likelihood.distribution.sqrtprec.T@x[idx_start:idx_end]) # Use adjoint function which excludes shift
                        idx_start = idx_end
                    out2 = L2.T @ x[idx_end:]
                    out  = out1 + out2                
                return out   
            self.M = M  
        else:
            raise TypeError("All likelihoods need to be callable or none need to be callable.") 

    @property
    def prior(self):
        return self.target.prior

    @property
    def likelihood(self):
        return self.target.likelihood
    
    @property
    def likelihoods(self):
        if isinstance(self.target, cuqi.distribution.Posterior):
            return [self.target.likelihood]
        elif isinstance(self.target, cuqi.distribution.MultipleLikelihoodPosterior):
            return self.target.likelihoods

    @property
    def model(self):
        return self.target.model     
    
    @property
    def models(self):
        if isinstance(self.target, cuqi.distribution.Posterior):
            return [self.target.model]
        elif isinstance(self.target, cuqi.distribution.MultipleLikelihoodPosterior):
            return self.target.models    

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
    
    def _check_posterior(self):
        # Check target type
        if not isinstance(self.target, (cuqi.distribution.Posterior, cuqi.distribution.MultipleLikelihoodPosterior)):
            raise ValueError(f"To initialize an object of type {self.__class__}, 'target' need to be of type 'cuqi.distribution.Posterior' or 'cuqi.distribution.MultipleLikelihoodPosterior'.")       

        # Check Linear model and Gaussian likelihood(s)
        if isinstance(self.target, cuqi.distribution.Posterior):
            if not isinstance(self.model, cuqi.model.AffineModel):
                raise TypeError("Model needs to be linear or affine")

            if not hasattr(self.likelihood.distribution, "sqrtprec"):
                raise TypeError("Distribution in Likelihood must contain a sqrtprec attribute")
            
        elif isinstance(self.target, cuqi.distribution.MultipleLikelihoodPosterior): # Elif used for further alternatives, e.g., stacked posterior
            for likelihood in self.likelihoods:
                if not isinstance(likelihood.model, cuqi.model.LinearModel):
                    raise TypeError("Model needs to be linear")

                if not hasattr(likelihood.distribution, "sqrtprec"):
                    raise TypeError("Distribution in Likelihood must contain a sqrtprec attribute")
        
        # Check Gaussian prior
        if not hasattr(self.prior, "sqrtprec"):
            raise TypeError("prior must contain a sqrtprec attribute")

        if not hasattr(self.prior, "sqrtprecTimesMean"):
            raise TypeError("Prior must contain a sqrtprecTimesMean attribute")


class RegularizedLinearRTO(LinearRTO):
    """
    Regularized Linear RTO (Randomize-Then-Optimize) sampler.

    Samples posterior related to the inverse problem with Gaussian likelihood and implicit Gaussian prior, and where the forward model is Linear.

    Parameters
    ------------
    target : `cuqi.distribution.Posterior`
        See `cuqi.sampler.LinearRTO`

    x0 : `np.ndarray` 
        Initial point for the sampler. *Optional*.

    maxit : int
        Maximum number of iterations of the inner FISTA solver. *Optional*.
        
    stepsize : string or float
        If stepsize is a string and equals either "automatic", then the stepsize is automatically estimated based on the spectral norm.
        If stepsize is a float, then this stepsize is used.

    abstol : float
        Absolute tolerance of the inner FISTA solver. *Optional*.

    callback : callable, *Optional*
        If set this function will be called after every sample.
        The signature of the callback function is `callback(sample, sample_index)`,
        where `sample` is the current sample and `sample_index` is the index of the sample.
        An example is shown in demos/demo31_callback.py.
        
    """
    def __init__(self, target, x0=None, maxit=100, stepsize = "automatic", abstol=1e-10, adaptive = True, **kwargs):

        if not callable(target.prior.proximal):
            raise TypeError("Projector needs to be callable")
        
        super().__init__(target, x0=x0, maxit=100, **kwargs)

        # Other parameters
        self.stepsize = stepsize
        self.abstol = abstol   
        self.adaptive = adaptive
        self.proximal = target.prior.proximal

    @property
    def prior(self):
        return self.target.prior.gaussian

    def _sample(self, N, Nb):   
        Ns = N+Nb   # number of simulations        
        samples = np.empty((self.n, Ns))
                   
        if isinstance(self.stepsize, str):
            if self.stepsize in ["automatic"]:
                if not callable(self.M):
                    M_op = scipyLinearOperator(self.M.shape, matvec = lambda v: self.M@v, rmatvec = lambda w: self.M.T@w)
                else:
                    M_op = scipyLinearOperator((len(self.b_tild), self.n), matvec = lambda v: self.M(v,1), rmatvec = lambda w: self.M(w,2))
                    
                _stepsize = 0.99/(estimate_spectral_norm(M_op)**2)
                # print(f"Estimated stepsize for regularized Linear RTO: {_stepsize}")
            else:
                raise ValueError("Stepsize choice not supported")
        else:
            _stepsize = self.stepsize
            
        # initial state   
        samples[:, 0] = self.x0
        for s in range(Ns-1):
            y = self.b_tild + np.random.randn(len(self.b_tild))
            sim = FISTA(self.M, y, self.proximal,
                        samples[:, s], maxit = self.maxit, stepsize = _stepsize, abstol = self.abstol, adaptive = self.adaptive)         
            samples[:, s+1], _ = sim.solve()
            
            self._print_progress(s+2,Ns) #s+2 is the sample number, s+1 is index assuming x0 is the first sample
            self._call_callback(samples[:, s+1], s+1)
        # remove burn-in
        samples = samples[:, Nb:]
        
        return samples, None, None


