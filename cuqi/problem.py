from warnings import WarningMessage
import cuqi
import numpy as np
import time


from cuqi.distribution import Cauchy_diff, GaussianCov, Laplace_diff, Gaussian, GMRF, Posterior
from cuqi.model import LinearModel, Model
from cuqi.geometry import _DefaultGeometry
from cuqi.utilities import ProblemInfo
from cuqi.pde import SteadyStateLinearPDE

class Generic(object):
    def __init__(self):
        raise NotImplementedError

class BayesianProblem(object):
    """
    Bayesian representation of inverse problem represented by likelihood and prior.

    Attributes
    ----------
    `likelihood: cuqi.likelihood.Likelihood`:
        summary: 'The likelihood function'
        example: model = cuqi.model.LinearModel(A);
                 likelihood = cuqi.distribution.Gaussian(model, std).to_likelihood(data) 
    `prior: cuqi.model.Distribution`:
        summary: 'A cuqi distribution for the prior'
        example: cuqi.distribution.Gaussian(mean, std, corrmat)
    `model: cuqi.model.Model`:
        summary: 'A cuqi forward model (optional)'
        example: cuqi.model.LinearModel(A) #A is a matrix

    Methods
    ----------
    `MAP()`:
        summary: 'Compute MAP estimate of the inverse problem.'
        NB: 'Requires the prior to be defined.'
    `Sample(Ns)`:
        summary: 'Sample Ns samples of the inverse problem.'
        NB: 'Requires the prior to be defined.'
    """
    @classmethod
    def get_components(cls, **kwargs):
        """
        Method that returns the model, the data and additional information to be used in formulating the Bayesian problem.
        
        Parameters:
        -----------
        Takes the same parameters that the corresponding class initializer takes. For example: :meth:`cuqi.testproblem.Deconvolution.get_components` takes the parameters of :meth:`cuqi.testproblem.Deconvolution` constructor. 
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
        return self.likelihood.data

    @property
    def likelihood(self):
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

    def ML(self):
        """Maximum Likelihood (ML) estimate"""
        x0 = np.random.randn(self.model.domain_dim)
        # Gradient should be used if available. We attempt to use gradients (in the
        # "try" part) and  if any error is encountered, it is caught and instead 
        # optimization without gradients is attempted.
        try: 
            print("Attempting to use gradients")
            gradfunc = lambda x: -self.likelihood.gradient(x)
            solver = cuqi.solver.minimize(
                                     lambda x: -self.likelihood.log(x), 
                                     x0, 
                                     gradfunc=gradfunc)
            x_BFGS, info_BFGS = solver.solve()
        except BaseException as err:
            print("Gradient not available, optimizing without.")
            solver = cuqi.solver.minimize(
                                     lambda x: -self.likelihood.log(x), 
                                     x0)
            x_BFGS, info_BFGS = solver.solve()
        return x_BFGS, info_BFGS


    def MAP(self):
        """MAP computed the MAP estimate of the posterior"""
        if self._check(Gaussian,Gaussian,LinearModel):
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
        else:
            x0 = np.random.randn(self.model.domain_dim)
            def posterior_logpdf(x):
                logpdf = -self.prior.logpdf(x) - self.likelihood.log(x)
                return logpdf
            # Gradient should be used if available. We attempt to use gradients (in the
            # "try" part) and  if any error is encountered, it is caught and instead 
            # optimization without gradients is attempted.
            try: 
                print("Attempting to use gradients")
                gradfunc = lambda x: -self.prior.gradient(x) - self.likelihood.gradient(x)
                solver = cuqi.solver.minimize(posterior_logpdf, 
                                              x0,
                                              gradfunc=gradfunc)
                x_BFGS, info_BFGS = solver.solve()
            except BaseException as err:
                print("Gradient not available, optimizing without.")        
                solver = cuqi.solver.minimize(posterior_logpdf, 
                                              x0)
                x_BFGS, info_BFGS = solver.solve()
            return x_BFGS, info_BFGS

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


    def sample_posterior(self,Ns):
        """Sample Ns samples of the posterior given data"""
        
        if self._check(Gaussian,Gaussian,LinearModel) and not self._check(Gaussian,GMRF) and self.model.domain_dim<=5000 and self.model.range_dim<=5000:
            print("Using direct sampling by Cholesky factor of inverse covariance. Only works for small-scale problems with dim<=5000.")
            return self._sampleMapCholesky(Ns)

        elif hasattr(self.prior,"sqrtprecTimesMean") and hasattr(self.likelihood.distribution,"sqrtprec") and isinstance(self.model,LinearModel):#self._check(GaussianCov,GaussianCov,LinearModel):
            print("Using Linear_RTO sampler")
            return self._sampleLinearRTO(Ns)

        elif self._check(Gaussian,Cauchy_diff) or self._check(Gaussian,Laplace_diff):
            print("Using Component-wise Metropolis-Hastings sampler")
            return self._sampleCWMH(Ns)
            
        elif self._check(Gaussian,Gaussian) or self._check(Gaussian,GMRF):
            print("Using preconditioned Crank-Nicolson sampler")
            return self._samplepCN(Ns)

        else:
            raise NotImplementedError(f'Sampler is not implemented for model: {type(self.model)}, likelihood: {type(self.likelihood.distribution)} and prior: {type(self.prior)}. Check documentation for available combinations.')

    def UQ(self,exact=None):
        print("Computing 5000 samples")
        samples = self.sample_posterior(5000)

        print("Plotting 95 percent confidence interval")
        if exact is not None:
            samples.plot_ci(95,exact=exact)
        elif hasattr(self,"exactSolution"):
            samples.plot_ci(95,exact=self.exactSolution)
        else:
            samples.plot_ci(95)

    def _check(self,distL,distP,typeModel=None):
        L = isinstance(self.likelihood.distribution,distL)
        P = isinstance(self.prior,distP)
        if typeModel is None:
            M = True
        else:
            M = isinstance(self.model,typeModel)
        return L and P and M
    def _sampleLinearRTO(self,Ns):
        posterior = Posterior(self.likelihood, self.prior)
        sampler = cuqi.sampler.Linear_RTO(posterior)
        return sampler.sample(Ns,0)

    def _sampleMapCholesky(self,Ns):
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

        x_map = self.MAP() #Compute MAP estimate
        C = np.linalg.inv(A.T@(np.linalg.inv(Ce)@A)+np.linalg.inv(Cx))
        L = np.linalg.cholesky(C)
        for s in range(Ns):
            x_s[:,s] = x_map.parameters + L@np.random.randn(n)
            # display iterations 
            if (s % 5e2) == 0:
                print("\r",'Sample', s, '/', Ns, end="")

        print("\r",'Sample', s+1, '/', Ns)
        print('Elapsed time:', time.time() - ti)
        
        return cuqi.samples.Samples(x_s,self.model.domain_geometry)
    
    def _sampleCWMH(self,Ns):
        # Dimension
        n = self.prior.dim
        
        # Set up target and proposal
        def target(x): return self.likelihood.log(x) + self.prior.logpdf(x)
        def proposal(x_t, sigma): return np.random.normal(x_t, sigma)

        # Set up sampler
        scale = 0.05*np.ones(n)
        x0 = 0.5*np.ones(n)
        MCMC = cuqi.sampler.CWMH(target, proposal, scale, x0)
        
        # Run sampler
        Nb = int(0.2*Ns)   # burn-in
        ti = time.time()
        x_s = MCMC.sample_adapt(Ns,Nb); #ToDo: Make results class
        print('Elapsed time:', time.time() - ti)
        
        return x_s

    def _samplepCN(self,Ns):
        # Dimension
        n = self.prior.dim
        
        # Set up target and proposal
        def target(x): return self.likelihood.log(x)
        #def proposal(ns): return self.prior.sample(ns)
        
        scale = 0.02
        #x0 = np.zeros(n)
        
        posterior = cuqi.distribution.Posterior(self.likelihood, self.prior)
        MCMC = cuqi.sampler.pCN(posterior,scale)      
        
        #TODO: Select burn-in 
        #Nb = int(0.25*Ns)   # burn-in

        #Run sampler
        ti = time.time()
        x_s = MCMC.sample_adapt(Ns,0) #ToDo: fix sampler input
        print('Elapsed time:', time.time() - ti)

        # Set geometry from prior
        #if hasattr(x_s,"geometry"):
        #    x_s.geometry = self.prior.geometry
        
        return x_s
