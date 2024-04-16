import scipy as sp
from scipy.linalg.interpolative import estimate_spectral_norm
from scipy.sparse.linalg import LinearOperator as scipyLinearOperator
import numpy as np
import cuqi
from cuqi.solver import CGLS, FISTA
from cuqi.experimental.mcmc import SamplerNew
from cuqi.array import CUQIarray

class UGLANew(SamplerNew):
    def __init__(self, target, initial_point=None, maxit=50, tol=1e-4, beta=1e-5, rng=None, **kwargs):

        # Other parameters
        self.maxit = maxit
        self.tol = tol
        self.beta = beta
        self.rng = rng

        super().__init__(target=target, initial_point=initial_point, **kwargs)

        if initial_point is None: #TODO: Replace later with a getter
            self.initial_point = np.zeros(self.dim)
            self._samples = [self.initial_point]

        self.current_point = self.initial_point
        self._acc = [1] # TODO. Check if we need this

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

    @SamplerNew.target.setter
    def target(self, value):
        """ Set the target density. Runs validation of the target. """
        # Accept tuple of inputs and construct posterior
        super(UGLANew, type(self)).target.fset(self, value)
        self._precompute()

    def _precompute(self):

        # Extract diff_op from target prior
        D = self.target.prior._diff_op
        n = D.shape[0]

        # Gaussian approximation of LMRF prior as function of x_k
        def Lk_fun(x_k):
            dd =  1/np.sqrt((D @ x_k)**2 + self.beta*np.ones(n))
            W = sp.sparse.diags(dd)
            return W.sqrt() @ D
        self.Lk_fun = Lk_fun

        # Now prepare "LinearRTO" type sampler. TODO: Use LinearRTO for this instead
        self._shift = 0

        # Pre-computations
        self._model = self.target.likelihood.model   
        self._data = self.target.likelihood.data
        self._m = len(self._data)
        self._L1 = self.target.likelihood.distribution.sqrtprec

        # If prior location is scalar, repeat it to match dimensions
        if len(self.target.prior.location) == 1:
            self._priorloc = np.repeat(self.target.prior.location, self.dim)
        else:
            self._priorloc = self.target.prior.location

        # Initial Laplace approx
        # self._L2 = Lk_fun(self.x0)
        self._L2 = Lk_fun(np.zeros(self.dim))
        self._L2mu = self._L2@self._priorloc
        self._b_tild = np.hstack([self._L1@self._data, self._L2mu]) 
        
        #self.n = len(self.x0)
        
        # Least squares form
        def M(x, flag):
            if flag == 1:
                out1 = self._L1 @ self._model.forward(x)
                out2 = np.sqrt(1/self.target.prior.scale)*(self._L2 @ x)
                out  = np.hstack([out1, out2])
            elif flag == 2:
                idx = int(self._m)
                out1 = self._model.adjoint(self._L1.T@x[:idx])
                out2 = np.sqrt(1/self.target.prior.scale)*(self._L2.T @ x[idx:])
                out  = out1 + out2                
            return out
        self.M = M

    def step(self):
        # Update Laplace approximation
        self._L2 = self.Lk_fun(self.current_point)
        self._L2mu = self._L2@self._priorloc
        self._b_tild = np.hstack([self._L1@self._data, self._L2mu]) 
    
        # Sample from approximate posterior
        e = cuqi.distribution.Normal(mean=np.zeros(len(self._b_tild)), std=1).sample(rng=self.rng)
        y = self._b_tild + e # Perturb data
        print(y)
        sim = CGLS(self.M, y, self.current_point, self.maxit, self.tol)                     
        self.current_point, _ = sim.solve()
        acc = 1
        return acc

    def tune(self, skip_len, update_count):
        pass
    
    def validate_target(self):
        # Check target type
        if not isinstance(self.target, cuqi.distribution.Posterior):
            raise ValueError(f"To initialize an object of type {self.__class__}, 'target' need to be of type 'cuqi.distribution.Posterior'.")       

        # Check Linear model
        if not isinstance(self.target.likelihood.model, cuqi.model.LinearModel):
            raise TypeError("Model needs to be linear")

        # Check Gaussian likelihood
        if not hasattr(self.target.likelihood.distribution, "sqrtprec"):
            raise TypeError("Distribution in Likelihood must contain a sqrtprec attribute")

        # Check that prior is LMRF
        if not isinstance(self.target.prior, cuqi.distribution.LMRF):
            raise ValueError('Unadjusted Gaussian Laplace approximation (UGLA) requires LMRF prior')
