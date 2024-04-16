import scipy as sp
import numpy as np
import cuqi
from cuqi.solver import CGLS
from cuqi.experimental.mcmc import SamplerNew

class UGLANew(SamplerNew):
    """ Unadjusted (Gaussian) Laplace Approximation sampler
    
    Samples an approximate posterior where the prior is approximated
    by a Gaussian distribution. The likelihood must be Gaussian.

    Currently only works for LMRF priors.

    The inner solver is Conjugate Gradient Least Squares (CGLS) solver.

    For more details see: Uribe, Felipe, et al. "A hybrid Gibbs sampler for edge-preserving 
    tomographic reconstruction with uncertain view angles." arXiv preprint arXiv:2104.06919 (2021).

    Parameters
    ----------
    target : `cuqi.distribution.Posterior`
        The target posterior distribution to sample.

    initial_point : ndarray
        Initial parameters. *Optional*

    maxit : int
        Maximum number of inner iterations for solver when generating one sample.

    tol : float
        Tolerance for inner solver. Will stop before maxit if the inner solvers convergence check reaches tol.

    beta : float
        Smoothing parameter for the Gaussian approximation of the Laplace distribution. Larger beta is easier to sample but is a worse approximation.

    callback : callable, *Optional*
        If set this function will be called after every sample.
        The signature of the callback function is `callback(sample, sample_index)`,
        where `sample` is the current sample and `sample_index` is the index of the sample.
        An example is shown in demos/demo31_callback.py.
    """
    def __init__(self, target, initial_point=None, maxit=50, tol=1e-4, beta=1e-5, **kwargs):

        # Parameters (beta is used in target setter)
        self.maxit = maxit
        self.tol = tol
        self.beta = beta

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
        D = self.prior._diff_op
        n = D.shape[0]

        # Gaussian approximation of LMRF prior as function of x_k
        def Lk_fun(x_k):
            dd =  1/np.sqrt((D @ x_k)**2 + self.beta*np.ones(n))
            W = sp.sparse.diags(dd)
            return W.sqrt() @ D
        self.Lk_fun = Lk_fun

        self._m = len(self.data)
        self._L1 = self.likelihood.distribution.sqrtprec

        # If prior location is scalar, repeat it to match dimensions
        if len(self.prior.location) == 1:
            self._priorloc = np.repeat(self.prior.location, self.dim)
        else:
            self._priorloc = self.prior.location

        # Initial Laplace approx
        # self._L2 = Lk_fun(self.x0)
        self._L2 = Lk_fun(np.zeros(self.dim)) #TODO: fix this
        self._L2mu = self._L2@self._priorloc
        self._b_tild = np.hstack([self._L1@self.data, self._L2mu]) 
        
        # Least squares form
        def M(x, flag):
            if flag == 1:
                out1 = self._L1 @ self.model.forward(x)
                out2 = np.sqrt(1/self.prior.scale)*(self._L2 @ x)
                out  = np.hstack([out1, out2])
            elif flag == 2:
                idx = int(self._m)
                out1 = self.model.adjoint(self._L1.T@x[:idx])
                out2 = np.sqrt(1/self.prior.scale)*(self._L2.T @ x[idx:])
                out  = out1 + out2                
            return out
        self.M = M

    def step(self):
        # Update Laplace approximation
        self._L2 = self.Lk_fun(self.current_point)
        self._L2mu = self._L2@self._priorloc
        self._b_tild = np.hstack([self._L1@self.data, self._L2mu]) 
    
        # Sample from approximate posterior
        e = np.random.randn(len(self._b_tild))
        y = self._b_tild + e # Perturb data
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
        if not isinstance(self.likelihood.model, cuqi.model.LinearModel):
            raise TypeError("Model needs to be linear")

        # Check Gaussian likelihood
        if not hasattr(self.likelihood.distribution, "sqrtprec"):
            raise TypeError("Distribution in Likelihood must contain a sqrtprec attribute")

        # Check that prior is LMRF
        if not isinstance(self.prior, cuqi.distribution.LMRF):
            raise ValueError('Unadjusted Gaussian Laplace approximation (UGLA) requires LMRF prior')
