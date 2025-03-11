import scipy as sp
from scipy.linalg.interpolative import estimate_spectral_norm
from scipy.sparse.linalg import LinearOperator as scipyLinearOperator
import numpy as np
import cuqi
from cuqi.solver import CGLS, FISTA, ADMM, ScipyLinearLSQ
from cuqi.experimental.mcmc import Sampler


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

    initial_point : `np.ndarray` 
        Initial point for the sampler. *Optional*.

    maxit : int
        Maximum number of iterations of the inner CGLS solver. *Optional*.

    tol : float
        Tolerance of the inner CGLS solver. *Optional*.

    callback : callable, optional
        A function that will be called after each sampling step. It can be useful for monitoring the sampler during sampling.
        The function should take three arguments: the sampler object, the index of the current sampling step, the total number of requested samples. The last two arguments are integers. An example of the callback function signature is: `callback(sampler, sample_index, num_of_samples)`.
        
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
    def models(self):
        if isinstance(self.target, cuqi.distribution.Posterior):
            return [self.target.model]
        elif isinstance(self.target, cuqi.distribution.MultipleLikelihoodPosterior):
            return self.target.models    

    def _precompute(self):
        L1 = [likelihood.distribution.sqrtprec for likelihood in self.likelihoods]
        L2 = self.prior.sqrtprec
        L2mu = self.prior.sqrtprecTimesMean

        # pre-computations
        self.n = self.prior.dim
        self.b_tild = np.hstack([L@(likelihood.data - model._shift) for (L, likelihood, model) in zip(L1, self.likelihoods, self.models)]+ [L2mu]) # With shift from AffineModel
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
                        out1 += likelihood.model._adjoint_func_no_shift(likelihood.distribution.sqrtprec.T@x[idx_start:idx_end])
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
            if not isinstance(self.model, cuqi.model.AffineModel):
                raise TypeError("Model needs to be linear or more generally affine")

            if not hasattr(self.likelihood.distribution, "sqrtprec"):
                raise TypeError("Distribution in Likelihood must contain a sqrtprec attribute")
            
        elif isinstance(self.target, cuqi.distribution.MultipleLikelihoodPosterior): # Elif used for further alternatives, e.g., stacked posterior
            for likelihood in self.likelihoods:
                if not isinstance(likelihood.model, cuqi.model.AffineModel):
                    raise TypeError("Model needs to be linear or more generally affine")

                if not hasattr(likelihood.distribution, "sqrtprec"):
                    raise TypeError("Distribution in Likelihood must contain a sqrtprec attribute")
        
        # Check Gaussian prior
        if not hasattr(self.prior, "sqrtprec"):
            raise TypeError("prior must contain a sqrtprec attribute")

        if not hasattr(self.prior, "sqrtprecTimesMean"):
            raise TypeError("Prior must contain a sqrtprecTimesMean attribute")
    
    def _get_default_initial_point(self, dim):
        """ Get the default initial point for the sampler. Defaults to an array of zeros. """
        return np.zeros(dim)

class RegularizedLinearRTO(LinearRTO):
    """
    Regularized Linear RTO (Randomize-Then-Optimize) sampler.

    Samples posterior related to the inverse problem with Gaussian likelihood and implicit Gaussian prior, and where the forward model is Linear.
    The sampler works by repeatedly solving regularized linear least squares problems for perturbed data.
    The solver for these optimization problems is chosen based on how the regularized is provided in the implicit Gaussian prior.
    Currently we use the following solvers:
    FISTA: [1] Beck, Amir, and Marc Teboulle. "A fast iterative shrinkage-thresholding algorithm for linear inverse problems." SIAM journal on imaging sciences 2.1 (2009): 183-202.
           Used when prior.proximal is callable.
    ADMM:  [2] Boyd et al. "Distributed optimization and statistical learning via the alternating direction method of multipliers."Foundations and Trends® in Machine learning, 2011.
           Used when prior.proximal is a list of penalty terms.
    ScipyLinearLSQ: Wrapper for Scipy's lsq_linear for the Trust Region Reflective algorithm. Optionally used when the constraint is either "nonnegativity" or "box".

    Parameters
    ------------
    target : `cuqi.distribution.Posterior`
        See `cuqi.sampler.LinearRTO`

    initial_point : `np.ndarray` 
        Initial point for the sampler. *Optional*.

    maxit : int
        Maximum number of iterations of the FISTA/ADMM/ScipyLinearLSQ solver. *Optional*.

    inner_max_it : int
        Maximum number of iterations of the CGLS solver used within the ADMM solver. *Optional*.
        
    stepsize : string or float
        If stepsize is a string and equals either "automatic", then the stepsize is automatically estimated based on the spectral norm.
        If stepsize is a float, then this stepsize is used.

    penalty_parameter : int
        Penalty parameter of the ADMM solver. *Optional*.
        See [2] or `cuqi.solver.ADMM`

    abstol : float
        Absolute tolerance of the FISTA/ScipyLinearLSQ solver. *Optional*.
    
    inner_abstol : float
        Tolerance parameter for ScipyLinearLSQ's inner solve of the unbounded least-squares problem. *Optional*.
    
    adaptive : bool
        If True, FISTA is used as solver, otherwise ISTA is used. *Optional*.
    
    solver : string
        If set to "ScipyLinearLSQ", solver is set to cuqi.solver.ScipyLinearLSQ, otherwise FISTA/ISTA or ADMM is used. Note "ScipyLinearLSQ" can only be used with `RegularizedGaussian` of `box` or `nonnegativity` constraint. *Optional*.

    callback : callable, optional
        A function that will be called after each sampling step. It can be useful for monitoring the sampler during sampling.
        The function should take three arguments: the sampler object, the index of the current sampling step, the total number of requested samples. The last two arguments are integers. An example of the callback function signature is: `callback(sampler, sample_index, num_of_samples)`.
        
    """
    def __init__(self, target=None, initial_point=None, maxit=100, inner_max_it=10, stepsize="automatic", penalty_parameter=10, abstol=1e-10, adaptive=True, solver=None, inner_abstol=None, **kwargs):
        
        super().__init__(target=target, initial_point=initial_point, **kwargs)

        # Other parameters
        self.stepsize = stepsize
        self.abstol = abstol
        self.inner_abstol = inner_abstol
        self.adaptive = adaptive
        self.maxit = maxit
        self.inner_max_it = inner_max_it
        self.penalty_parameter = penalty_parameter
        self.solver = solver

    def _initialize(self):
        super()._initialize()
        if self.solver is None:
            self.solver = "FISTA" if callable(self.proximal) else "ADMM"
        if self.solver == "FISTA":
            self._stepsize = self._choose_stepsize()

    @property
    def solver(self):
        return self._solver

    @solver.setter
    def solver(self, value):
        if value == "ScipyLinearLSQ":
            if (self.target.prior.preset["constraint"] == "nonnegativity" or self.target.prior.preset["constraint"] == "box"):
                self._solver = value
            else:
                raise ValueError("ScipyLinearLSQ only supports RegularizedGaussian with box or nonnegativity constraint.")
        else:
            self._solver = value

    @property
    def proximal(self):
        return self.target.prior.proximal
    
    def validate_target(self):
        super().validate_target()
        if not isinstance(self.target.prior, (cuqi.implicitprior.RegularizedGaussian, cuqi.implicitprior.RegularizedGMRF)):
            raise TypeError("Prior needs to be RegularizedGaussian or RegularizedGMRF")

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

        if self.solver == "FISTA":
            sim = FISTA(self.M, y, self.proximal,
                        self.current_point, maxit = self.maxit, stepsize = self._stepsize, abstol = self.abstol, adaptive = self.adaptive)         
        elif self.solver == "ADMM":
            sim = ADMM(self.M, y, self.proximal,
                        self.current_point, self.penalty_parameter, maxit = self.maxit, inner_max_it = self.inner_max_it, adaptive = self.adaptive)
        elif self.solver == "ScipyLinearLSQ":
                A_op = sp.sparse.linalg.LinearOperator((sum([llh.dim for llh in self.likelihoods])+self.target.prior.dim, self.target.prior.dim),
                                        matvec=lambda x: self.M(x, 1),
                                        rmatvec=lambda x: self.M(x, 2)
                                        )
                sim = ScipyLinearLSQ(A_op, y, self.target.prior._box_bounds, 
                                     max_iter = self.maxit,
                                     lsmr_maxiter = self.inner_max_it, 
                                     tol = self.abstol,
                                     lsmr_tol = self.inner_abstol)
        else:
            raise ValueError("Choice of solver not supported.")

        self.current_point, _ = sim.solve()
        acc = 1
        return acc