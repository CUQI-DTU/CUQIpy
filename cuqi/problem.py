import numpy as np
import time

import cuqi
from cuqi import config
from cuqi.distribution import Cauchy_diff, GaussianCov, InverseGamma, Laplace_diff, Gaussian, GMRF, Lognormal, Posterior, LMRF, Laplace, Beta
from cuqi.model import LinearModel, Model
from cuqi.geometry import _DefaultGeometry
from cuqi.utilities import ProblemInfo
from cuqi.pde import SteadyStateLinearPDE

class BayesianProblem(object):
    """Representation of a Bayesian inverse problem (posterior) defined by a likelihood and prior.
    
    .. math::

        \pi_\mathrm{posterior}(\mathbf{x} \mid \mathbf{b}) \propto \pi_\mathrm{likelihood}(\mathbf{b} \mid \mathbf{x}) \pi_\mathrm{prior}(\mathbf{x}),

    where :math:`\pi_\mathrm{Likelihood}(\mathbf{b} \mid \mathbf{x})` is a :class:`cuqi.likelihood.Likelihood` function and :math:`\pi_\mathrm{prior}(\mathbf{x})` is a :class:`cuqi.distribution.Distribution`.

    The main goal of this class is to provide fully automatic methods for computing samples or point estimates of the posterior distribution.

    Parameters
    ----------
    likelihood : Likelihood
        The likelihood function.

    prior : Distribution
        The prior distribution.

    Attributes
    ----------
    likelihood: Likelihood
        The likelihood function.

    prior: Distribution
        The prior distribution.

    posterior: Distribution
        The posterior distribution (inferred from likelihood and prior).

    model: Model
        The deterministic model for the inverse problem (inferred from likelihood).

    data: CUQIarray
        The observed data (inferred from likelihood).

    Methods
    -------
    sample_posterior(Ns):
        Sample Ns samples of the posterior.
    MAP():
        Compute Maximum a posteriori (MAP) estimate of the posterior.
    ML():
        Compute maximum likelihood estimate.
    """
    @classmethod
    def get_components(cls, **kwargs):
        """
        Method that returns the model, the data and additional information to be used in formulating the Bayesian problem.
        
        Parameters:
        -----------
        Takes the same parameters that the corresponding class initializer takes. For example: :meth:`cuqi.testproblem.Deconvolution1D.get_components` takes the parameters of :meth:`cuqi.testproblem.Deconvolution1D` constructor. 
        """

        problem_info = ProblemInfo() #Instead of a dict, we use our ProblemInfo dataclass.
        problem = cls(**kwargs)

        for key, value in vars(problem_info).items():
            if hasattr(problem, key):
                setattr(problem_info,key,vars(problem)[key])

        return problem.model, problem.data, problem_info

    def __init__(self,likelihood,prior):
        self.likelihood = likelihood
        self.prior = prior

    @property
    def data(self):
        """Extract the observed data from likelihood"""
        return self.likelihood.data

    @property
    def likelihood(self):
        """The likelihood function."""
        return self._likelihood
    
    @likelihood.setter
    def likelihood(self, value):
        self._likelihood = value
        if value is not None:        
            msg = f"{self.model.__class__} range_geometry and likelihood data distribution geometry are not consistent"
            self.likelihood.distribution.geometry,self.model.range_geometry = \
                self._check_geometries_consistency(self.likelihood.distribution.geometry,self.model.range_geometry,msg)
            if hasattr(self,'prior'):
                self.prior=self.prior

    @property
    def prior(self):
        """The prior distribution"""
        return self._prior
    
    @prior.setter
    def prior(self, value):
        self._prior = value
        if value is not None and self.model is not None:
            msg = f"{self.model.__class__} domain_geometry and prior geometry are not consistent"
            self.prior.geometry,self.model.domain_geometry = \
                self._check_geometries_consistency(self.prior.geometry,self.model.domain_geometry,msg)

    @property
    def model(self):
        """Extract the cuqi model from likelihood."""
        return self.likelihood.model

    @property
    def posterior(self):
        """Create posterior distribution from likelihood and prior"""
        return Posterior(self.likelihood, self.prior)

    def ML(self):
        """Maximum Likelihood (ML) estimate"""
        # Print warning to user about the automatic solver selection
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! Automatic solver selection is experimental. !!!")
        print("!!!    Always validate the computed results.    !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("")
        print("Using scipy.optimize.minimize on negative log-likelihood")
        print("x0: random vector")
        x0 = np.random.randn(self.model.domain_dim)

        if self._check_posterior(must_have_gradient=True):
            print("Optimizing with exact gradients")
            gradfunc = lambda x: -self.likelihood.gradient(x)
            solver = cuqi.solver.minimize(
                                     lambda x: -self.likelihood.log(x), 
                                     x0, 
                                     gradfunc=gradfunc)
            x_BFGS, info_BFGS = solver.solve()
        else:
            print("Optimizing with approximate gradients.")
            solver = cuqi.solver.minimize(
                                     lambda x: -self.likelihood.log(x), 
                                     x0)
            x_BFGS, info_BFGS = solver.solve()
        return x_BFGS, info_BFGS


    def MAP(self, disp=True):
        """Compute the MAP estimate of the posterior.
        
        Parameters
        ----------

        disp : bool
            display info messages? (True or False).
        
        """

        if disp:
            # Print warning to user about the automatic solver selection
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("!!! Automatic solver selection is experimental. !!!")
            print("!!!    Always validate the computed results.    !!!")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("")

        if self._check_posterior((Gaussian, GaussianCov), Gaussian, LinearModel, max_dim=config.MAX_DIM_INV):
            if disp: print(f"Using direct MAP of Gaussian posterior. Only works for small-scale problems with dim<={config.MAX_DIM_INV}.")
            b  = self.data
            A  = self.model.get_matrix()
            Ce = self.likelihood.distribution.Sigma
            x0 = self.prior.mean
            Cx = self.prior.Sigma

            #Basic MAP estimate using closed-form expression Tarantola 2005 (3.37-3.38)
            rhs = b-A@x0
            sysm = A@Cx@A.T+Ce
            map_estimate = x0 + Cx@(A.T@np.linalg.solve(sysm,rhs))
            return cuqi.samples.CUQIarray(map_estimate, geometry=self.model.domain_geometry)

        # If no specific implementation exists, use numerical optimization.
        if disp: print("Using scipy.optimize.minimize on negative logpdf of posterior")
        if disp: print("x0: random vector")
        x0 = np.random.randn(self.model.domain_dim)
        def posterior_logpdf(x):
            return -self.posterior.logpdf(x)

        if self._check_posterior(must_have_gradient=True):
            if disp: print("Optimizing with exact gradients")
            gradfunc = lambda x: -self.posterior.gradient(x)
            solver = cuqi.solver.minimize(posterior_logpdf, 
                                            x0,
                                            gradfunc=gradfunc)
            x_BFGS, info_BFGS = solver.solve()
        else:
            if disp: print("Optimizing with approximate gradients.")      
            solver = cuqi.solver.minimize(posterior_logpdf, 
                                            x0)
            x_BFGS, info_BFGS = solver.solve()
        return x_BFGS, info_BFGS

    def sample_posterior(self, Ns, callback=None) -> cuqi.samples.Samples:
        """Sample the posterior. Sampler choice and tuning is handled automatically.
        
        Parameters
        ----------
        Ns : int
            Number of samples to draw.

        callback : callable, *Optional*
            If set this function will be called after every sample.
            The signature of the callback function is `callback(sample, sample_index)`,
            where `sample` is the current sample and `sample_index` is the index of the sample.
            An example is shown in demos/demo31_callback.py.

        Returns
        -------
        samples : cuqi.samples.Samples
            Samples from the posterior.
        
        """

        # Print warning to user about the automatic sampler selection
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! Automatic sampler selection is experimental. !!!")
        print("!!!    Always validate the computed results.     !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("")

        # For Gaussian small-scale we can use direct sampling
        if self._check_posterior((Gaussian, GaussianCov), (Gaussian, GaussianCov), LinearModel, config.MAX_DIM_INV) and not self._check_posterior(GMRF):
            return self._sampleMapCholesky(Ns, callback)

        # For larger-scale Gaussian we use Linear RTO. TODO: Improve checking once we have a common Gaussian class.
        elif hasattr(self.prior,"sqrtprecTimesMean") and hasattr(self.likelihood.distribution,"sqrtprec") and isinstance(self.model,LinearModel):
            return self._sampleLinearRTO(Ns, callback)

        # For Laplace_diff we use our awesome unadjusted Laplace approximation!
        elif self._check_posterior(Laplace_diff, (Gaussian, GaussianCov)):
            return self._sampleUnadjustedLaplaceApproximation(Ns, callback)

        # If we have gradients, use NUTS!
        # TODO: Fix cases where we have gradients but NUTS fails (see checks)
        elif self._check_posterior(must_have_gradient=True) and not self._check_posterior((Beta, InverseGamma, Lognormal)):
            return self._sampleNUTS(Ns, callback)

        # For Gaussians with non-linear model we use pCN
        elif self._check_posterior((Gaussian, GMRF, GaussianCov), (Gaussian, GaussianCov)):
            return self._samplepCN(Ns, callback)

        # For the remainder of valid cases we use CWMH
        elif self._check_posterior(LMRF):
            return self._sampleCWMH(Ns, callback)

        else:
            raise NotImplementedError(f"Automatic sampler choice is not implemented for model: {type(self.model)}, likelihood: {type(self.likelihood.distribution)} and prior: {type(self.prior)} and dim {self.prior.dim}. Manual sampler choice can be done via the 'sampler' module. Posterior distribution can be extracted via '.posterior' of any testproblem (BayesianProblem).")

    def UQ(self, exact=None):
        print("Computing 5000 samples")
        samples = self.sample_posterior(5000)

        print("Plotting 95 percent credibility interval")
        if exact is not None:
            samples.plot_ci(95,exact=exact)
        elif hasattr(self,"exactSolution"):
            samples.plot_ci(95,exact=self.exactSolution)
        else:
            samples.plot_ci(95)

    def _sampleLinearRTO(self,Ns, callback=None):
        print("Using Linear_RTO sampler.")
        print("burn-in: 20%")

        # Start timing
        ti = time.time()

        # Sample
        Nb = int(0.2*Ns)   # burn-in
        sampler = cuqi.sampler.Linear_RTO(self.posterior, callback=callback)
        samples = sampler.sample(Ns, Nb)

        # Print timing
        print('Elapsed time:', time.time() - ti)

        return samples

    def _sampleMapCholesky(self, Ns, callback=None):
        print(f"Using direct sampling of Gaussian posterior. Only works for small-scale problems with dim<={config.MAX_DIM_INV}.")
        # Start timing
        ti = time.time()

        b  = self.data
        A  = self.model.get_matrix()
        Ce = self.likelihood.distribution.Sigma
        x0 = self.prior.mean
        Cx = self.prior.Sigma

        # Preallocate samples
        n = self.prior.dim 
        x_s = np.zeros((n,Ns))

        x_map = self.MAP(disp=False) #Compute MAP estimate
        C = np.linalg.inv(A.T@(np.linalg.inv(Ce)@A)+np.linalg.inv(Cx))
        L = np.linalg.cholesky(C)
        for s in range(Ns):
            x_s[:,s] = x_map.parameters + L@np.random.randn(n)
            # display iterations 
            if (s % 5e2) == 0:
                print("\r",'Sample', s, '/', Ns, end="")
            
            # Callback function
            if callback is not None:
                callback(x_s[:,s], s)

        print("\r",'Sample', s+1, '/', Ns)
        print('Elapsed time:', time.time() - ti)
        
        return cuqi.samples.Samples(x_s,self.model.domain_geometry)
    
    def _sampleCWMH(self, Ns, callback=None):
        print("Using Component-wise Metropolis-Hastings (CWMH) sampler (sample_adapt)")
        print("burn-in: 20%, scale: 0.05, x0: 0.5 (vector)")

        # Dimension
        n = self.prior.dim
        
        # Set up target and proposal
        def proposal(x_t, sigma): return np.random.normal(x_t, sigma)

        # Set up sampler
        scale = 0.05*np.ones(n)
        x0 = 0.5*np.ones(n)
        MCMC = cuqi.sampler.CWMH(self.posterior, proposal, scale, x0, callback=callback)
        
        # Run sampler
        Nb = int(0.2*Ns)   # burn-in
        ti = time.time()
        x_s = MCMC.sample_adapt(Ns,Nb); #ToDo: Make results class
        print('Elapsed time:', time.time() - ti)
        
        return x_s

    def _samplepCN(self, Ns, callback=None):
        print("Using preconditioned Crank-Nicolson (pCN) sampler (sample_adapt)")
        print("burn-in: 20%, scale: 0.02")

        scale = 0.02
        #x0 = np.zeros(n)
        
        MCMC = cuqi.sampler.pCN(self.posterior, scale, callback=callback)      
        
        #Run sampler
        Nb = int(0.2*Ns)
        ti = time.time()
        x_s = MCMC.sample_adapt(Ns, Nb)
        print('Elapsed time:', time.time() - ti)
       
        return x_s

    def _sampleNUTS(self, Ns, callback=None):
        print("Using No-U-Turn (NUTS) sampler")
        print("burn-in: 20%")

        # MAP
        #print("Computing MAP ESTIMATE")
        #x_map, _ = self.MAP()
        
        MCMC = cuqi.sampler.NUTS(self.posterior, callback=callback)
        
        # Run sampler
        Nb = int(0.2*Ns)   # burn-in
        ti = time.time()
        x_s = MCMC.sample_adapt(Ns+Nb); # TODO. FIX burn-in for NUTS!
        x_s = x_s.burnthin(Nb)
        print('Elapsed time:', time.time() - ti)
        
        return x_s

    def _sampleUnadjustedLaplaceApproximation(self, Ns, callback=None):
        print("Using Unadjusted Laplace Approximation sampler")
        print("burn-in: 20%")

        # Start timing
        ti = time.time()

        # Sample
        Nb = int(0.2*Ns)
        sampler = cuqi.sampler.UnadjustedLaplaceApproximation(self.posterior, callback=callback)
        samples = sampler.sample(Ns, Nb)

        # Print timing
        print('Elapsed time:', time.time() - ti)

        return samples

    def _check_geometries_consistency(self, geom1, geom2, fail_msg):
        """checks geom1 and geom2 consistency . If both are of type `_DefaultGeometry` they need to be equal. If one of them is of `_DefaultGeometry` type, it will take the value of the other one. If both of them are user defined, they need to be consistent"""
        if isinstance(geom1,_DefaultGeometry):
            if isinstance(geom2,_DefaultGeometry):
                if geom1 == geom2:
                    return geom1,geom2
            else: 
                return geom2, geom2
        else:
            if isinstance(geom2,_DefaultGeometry):
                return geom1,geom1
            else:
                if geom1 == geom2:
                    return geom1,geom2
        raise Exception(fail_msg)

    def _check_posterior(self, prior_type=None, likelihood_type=None, model_type=None, max_dim=None, must_have_gradient=False):
        """Returns true if components of the posterior reflects the types (can be tuple of types) given as input."""
        # Prior check
        if prior_type is None:
            P = True
        else:
            P = isinstance(self.prior, prior_type)

        # Likelihood check
        if likelihood_type is None:
            L = True
        else:
            L = isinstance(self.likelihood.distribution, likelihood_type)

        # Model check
        if model_type is None:
            M = True
        else:
            M = isinstance(self.model, model_type)

        #Dimension check
        if max_dim is None:
            D = True
        else:
            D = self.model.domain_dim<=max_dim and self.model.range_dim<=max_dim

        # Require gradient?
        if must_have_gradient:
            try: 
                self.prior.gradient(np.zeros(self.prior.dim))
                self.likelihood.gradient(np.zeros(self.likelihood.dim))
                G = True
            except (NotImplementedError, AttributeError):
                G = False
        else:
            G = True

        return L and P and M and D and G