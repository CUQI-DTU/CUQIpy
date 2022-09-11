import numpy as np
import time

import cuqi
from cuqi import config
from cuqi.distribution import Cauchy_diff, GaussianCov, InverseGamma, Laplace_diff, Gaussian, GMRF, Lognormal, Posterior, LMRF, Laplace, Beta, JointDistribution
from cuqi.likelihood import Likelihood
from cuqi.model import LinearModel, Model
from cuqi.geometry import _DefaultGeometry
from cuqi.utilities import ProblemInfo
from cuqi.pde import SteadyStateLinearPDE

from copy import copy


class BayesianProblem(object):
    """Representation of a Bayesian inverse problem (posterior) defined by a likelihood and prior.
    
    .. math::

        \pi_\mathrm{posterior}(\mathbf{x} \mid \mathbf{b}) \propto \pi_\mathrm{likelihood}(\mathbf{b} \mid \mathbf{x}) \pi_\mathrm{prior}(\mathbf{x}),

    where :math:`\pi_\mathrm{Likelihood}(\mathbf{b} \mid \mathbf{x})` is a :class:`cuqi.likelihood.Likelihood` function and :math:`\pi_\mathrm{prior}(\mathbf{x})` is a :class:`cuqi.distribution.Distribution`.

    The main goal of this class is to provide fully automatic methods for computing samples or point estimates of the posterior distribution.

    Parameters
    ----------
    *densities
        The densities representing the problem

    **observations
        The observations.

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

    def __init__(self, *densities, **observations):
        self._target = JointDistribution(*densities)(**observations)

    def set_data(self, **kwargs):
        """ Set the data of the problem. This conditions the underlying joint distribution on the data. """
        if not isinstance(self._target, JointDistribution):
            raise ValueError("Unable to set data for this problem. Maybe data is already set?")
        self._target = self._target(**kwargs)
        return self

    @property
    def data(self):
        """Extract the observed data from likelihood"""
        return self.likelihood.data

    @property
    def likelihood(self):
        """The likelihood function."""
        if not isinstance(self._target, Posterior):
            raise ValueError(f"Unable to extract likelihood from this problem. Current target is: \n {self._target}")
        return self._target.likelihood

    @likelihood.setter
    def likelihood(self, likelihood):
        if not isinstance(self._target, Posterior):
            raise ValueError(f"Unable to set likelihood for this problem. Current target is: \n {self._target}")
        self._target.likelihood = likelihood

    @property
    def prior(self):
        """The prior distribution"""
        if not isinstance(self._target, Posterior):
            raise ValueError(f"Unable to extract prior from this problem. Current target is: \n {self._target}")
        return self._target.prior

    @prior.setter
    def prior(self, prior):
        if not isinstance(self._target, Posterior):
            raise ValueError(f"Unable to set prior for this problem. Current target is: \n {self._target}")
        self._target.prior = prior

    @property
    def model(self):
        """Extract the cuqi model from likelihood."""
        return self.likelihood.model

    @property
    def posterior(self):
        """Create posterior distribution from likelihood and prior"""
        if not isinstance(self._target, Posterior):
            raise ValueError(f"Unable to extract posterior for this problem. Current target is: \n {self._target}")
        return self._target

    def ML(self, disp=True, x0=None):
        """ Compute the Maximum Likelihood (ML) estimate of the posterior.
        
        Parameters
        ----------

        disp : bool
            display info messages? (True or False).

        x0 : CUQIarray or ndarray
            User-specified initial guess for the solver. Defaults to a ones vector.

        Returns
        -------
        CUQIarray
            ML estimate of the posterior. Solver info is stored in the returned CUQIarray `info` attribute.
        
        """
        if disp:
            # Print warning to user about the automatic solver selection
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("!!! Automatic solver selection is experimental. !!!")
            print("!!!    Always validate the computed results.    !!!")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("")

        x_ML, solver_info = self._solve_max_point(self.likelihood, disp=disp, x0=x0)

        # Wrap the result in a CUQIarray and add solver info
        x_ML = cuqi.samples.CUQIarray(x_ML, geometry=self.likelihood.geometry)
        x_ML.info = solver_info

        return x_ML


    def MAP(self, disp=True, x0=None):
        """ Compute the Maximum A Posteriori (MAP) estimate of the posterior.
        
        Parameters
        ----------

        disp : bool
            display info messages? (True or False).

        x0 : CUQIarray or ndarray
            User-specified initial guess for the solver. Defaults to a ones vector.

        Returns
        -------
        CUQIarray
            MAP estimate of the posterior. Solver info is stored in the returned CUQIarray `info` attribute.
        
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
            x_MAP = x0 + Cx@(A.T@np.linalg.solve(sysm,rhs))
            solver_info = {"solver": "direct"}

        else: # If no specific implementation exists, use numerical optimization.
            x_MAP, solver_info = self._solve_max_point(self.posterior, disp=disp, x0=x0)

        # Wrap the result in a CUQIarray and add solver info
        x_MAP = cuqi.samples.CUQIarray(x_MAP, geometry=self.posterior.geometry)
        x_MAP.info = solver_info
        return x_MAP

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

    def sample_prior(self, Ns, callback=None) -> cuqi.samples.Samples:
        """ Sample the prior distribution. Sampler choice and tuning is handled automatically. """

        # Try sampling prior directly
        try:
            return self.prior.sample(Ns)
        except NotImplementedError:
            pass

        # If no direct method exists redefine posterior to one with a constant likelihood and sample from posterior
        print("Using MCMC methods from sampler module to sample prior.")
        print("Make sure enough samples are drawn for convergence.")
        print("")

        # Create a copy of self
        prior_problem = copy(self)

        # Set likelihood to constant
        model = cuqi.model.LinearModel(lambda x: 0*x, lambda y: 0*y, self.model.range_geometry, self.model.domain_geometry)
        likelihood = cuqi.distribution.GaussianCov(model, 1).to_likelihood(np.zeros(self.model.range_dim)) # p(y|x)=constant
        prior_problem.likelihood = likelihood

        # Now sample prior problem
        return prior_problem.sample_posterior(Ns, callback)

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
        x_s = MCMC.sample_adapt(Ns,Nb)
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

    def _solve_max_point(self, density, disp=True, x0=None):
        """ This is a helper function for point estimation of the maximum of a density (e.g. posterior or likelihood) using solver module.
        
        Parameters
        ----------
        density : Density (Distribution or Likelihood)
            The density or likelihood to compute the maximum point of the negative log.

        disp : bool
            display info messages? (True or False).
        """

        if disp: print(f"Using scipy.optimize.fmin_l_bfgs_b on negative log of {density.__class__.__name__}")
        if disp: print("x0: ones vector")
        
        # Get the function to minimize (negative log-likelihood or negative log-posterior)
        def func(x): return -density.logd(x)

        # Initial value if not given
        if x0 is None:
            x0 = np.ones(self.model.domain_dim)

        # Get the gradient (if available)
        try: 
            density.gradient(x0)
            def gradfunc(x): return -density.gradient(x)
            if disp: print("Optimizing with exact gradients")
        except (NotImplementedError, AttributeError):
            gradfunc = None
            if disp: print("Optimizing with approximate gradients.") 

        # Compute point estimate
        solver = cuqi.solver.L_BFGS_B(func, x0, gradfunc=gradfunc)
        x_MAP, solver_info = solver.solve()

        # Add info on solver choice
        solver_info["solver"] = "L-BFGS-B"

        return x_MAP, solver_info

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
