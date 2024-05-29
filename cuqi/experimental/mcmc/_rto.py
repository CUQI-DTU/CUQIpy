import scipy as sp
from scipy.linalg.interpolative import estimate_spectral_norm
from scipy.sparse.linalg import LinearOperator as scipyLinearOperator
import numpy as np
import cuqi
from cuqi.solver import CGLS, FISTA
from cuqi.experimental.mcmc import SamplerNew


class LinearRTONew(SamplerNew):
    """
    Linear RTO (Randomize-Then-Optimize) sampler.

    Samples posterior related to the inverse problem with Gaussian likelihood and prior, and where the forward model is Linear.

    Parameters
    ------------
    target : `cuqi.distribution.Posterior`, `cuqi.distribution.MultipleLikelihoodPosterior` or 5-dimensional tuple.
        If target is of type cuqi.distribution.Posterior or cuqi.distribution.MultipleLikelihoodPosterior, it represents the posterior distribution.
        If target is a 5-dimensional tuple, it assumes the following structure:
        (data, model, L_sqrtprec, P_mean, P_sqrtrec)
        
        Here:
        data: is a m-dimensional numpy array containing the measured data.
        model: is a m by n dimensional matrix or LinearModel representing the forward model.
        L_sqrtprec: is the squareroot of the precision matrix of the Gaussian likelihood.
        P_mean: is the prior mean.
        P_sqrtprec: is the squareroot of the precision matrix of the Gaussian mean.

    initial_point : `np.ndarray` 
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
    def __init__(self, target=None, initial_point=None, maxit=10, tol=1e-6, **kwargs):

        super().__init__(target=target, initial_point=initial_point, **kwargs)

        # Other parameters
        self.maxit = maxit
        self.tol = tol

    def _initialize(self):
        self._precompute()

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
    def data(self):
        return self.target.data
    
    def _precompute(self):
        L1 = [likelihood.distribution.sqrtprec for likelihood in self.likelihoods]
        L2 = self.prior.sqrtprec
        L2mu = self.prior.sqrtprecTimesMean

        # pre-computations
        self.n = self.prior.dim
        self.b_tild = np.hstack([L@likelihood.data for (L, likelihood) in zip(L1, self.likelihoods)]+ [L2mu]) 

        callability = [callable(likelihood.model) for likelihood in self.likelihoods]
        notcallability = [not c for c in callability]
        if all(notcallability):
            self.M = sp.sparse.vstack([L@likelihood.model for (L, likelihood) in zip(L1, self.likelihoods)] + [L2])
        elif all(callability):
            # in this case, model is a function doing forward and backward operations
            def M(x, flag):
                if flag == 1:
                    out1 = [L @ likelihood.model.forward(x) for (L, likelihood) in zip(L1, self.likelihoods)]
                    out2 = L2 @ x
                    out  = np.hstack(out1 + [out2])
                elif flag == 2:
                    idx_start = 0
                    idx_end = 0
                    out1 = np.zeros(self.n)
                    for likelihood in self.likelihoods:
                        idx_end += len(likelihood.data)
                        out1 += likelihood.model.adjoint(likelihood.distribution.sqrtprec.T@x[idx_start:idx_end])
                        idx_start = idx_end
                    out2 = L2.T @ x[idx_end:]
                    out  = out1 + out2                
                return out   
            self.M = M  
        else:
            raise TypeError("All likelihoods need to be callable or none need to be callable.")

    def step(self):
        y = self.b_tild + np.random.randn(len(self.b_tild))
        sim = CGLS(self.M, y, self.current_point, self.maxit, self.tol)            
        self.current_point, _ = sim.solve()
        acc = 1
        return acc

    def tune(self, skip_len, update_count):
        pass
    
    def validate_target(self):
        # Check target type
        if not isinstance(self.target, (cuqi.distribution.Posterior, cuqi.distribution.MultipleLikelihoodPosterior)):
            raise ValueError(f"To initialize an object of type {self.__class__}, 'target' need to be of type 'cuqi.distribution.Posterior' or 'cuqi.distribution.MultipleLikelihoodPosterior'.")       

        # Check Linear model and Gaussian likelihood(s)
        if isinstance(self.target, cuqi.distribution.Posterior):
            if not isinstance(self.model, cuqi.model.LinearModel):
                raise TypeError("Model needs to be linear")

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
    
    @property
    def _default_initial_point(self):
        """ Get the default initial point for the sampler. Defaults to an array of zeros. """
        return np.zeros(self.dim)

class RegularizedLinearRTONew(LinearRTONew):
    """
    Regularized Linear RTO (Randomize-Then-Optimize) sampler.

    Samples posterior related to the inverse problem with Gaussian likelihood and implicit Gaussian prior, and where the forward model is Linear.

    Parameters
    ------------
    target : `cuqi.distribution.Posterior`
        See `cuqi.sampler.LinearRTO`

    initial_point : `np.ndarray` 
        Initial point for the sampler. *Optional*.

    maxit : int
        Maximum number of iterations of the inner FISTA solver. *Optional*.
        
    stepsize : string or float
        If stepsize is a string and equals either "automatic", then the stepsize is automatically estimated based on the spectral norm.
        If stepsize is a float, then this stepsize is used.

    abstol : float
        Absolute tolerance of the inner FISTA solver. *Optional*.
    
    adaptive : bool
        If True, FISTA is used as inner solver, otherwise ISTA is used. *Optional*.

    callback : callable, *Optional*
        If set this function will be called after every sample.
        The signature of the callback function is `callback(sample, sample_index)`,
        where `sample` is the current sample and `sample_index` is the index of the sample.
        An example is shown in demos/demo31_callback.py.
        
    """
    def __init__(self, target=None, initial_point=None, maxit=100, stepsize="automatic", abstol=1e-10, adaptive=True, **kwargs):
        
        super().__init__(target=target, initial_point=initial_point, **kwargs)

        # Other parameters
        self.stepsize = stepsize
        self.abstol = abstol   
        self.adaptive = adaptive
        self.maxit = maxit

    def _initialize(self):
        super()._initialize()
        self._stepsize = self._choose_stepsize()

    @property
    def proximal(self):
        return self.target.prior.proximal
    
    def validate_target(self):
        super().validate_target()
        if not isinstance(self.target.prior, (cuqi.implicitprior.RegularizedGaussian, cuqi.implicitprior.RegularizedGMRF)):
            raise TypeError("Prior needs to be RegularizedGaussian or RegularizedGMRF")
        if not callable(self.proximal):
            raise TypeError("Proximal needs to be callable")

    def _choose_stepsize(self):
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
        return _stepsize

    @property
    def prior(self):
        return self.target.prior.gaussian

    def step(self):
        y = self.b_tild + np.random.randn(len(self.b_tild))
        sim = FISTA(self.M, y, self.current_point, self.proximal,
                    maxit = self.maxit, stepsize = self._stepsize, abstol = self.abstol, adaptive = self.adaptive)         
        self.current_point, _ = sim.solve()
        acc = 1
        return acc
