import scipy as sp
import numpy as np
import cuqi
from cuqi.solver import CGLS
from cuqi.experimental.mcmc import Sampler

class UGLA(Sampler):
    """ Unadjusted (Gaussian) Laplace Approximation sampler
    
    Samples an approximate posterior where the prior is approximated
    by a Gaussian distribution. The likelihood must be Gaussian.

    Currently only works for LMRF priors.

    The inner solver is Conjugate Gradient Least Squares (CGLS) solver.

    For more details see: Uribe, Felipe, et al. A hybrid Gibbs sampler for edge-preserving 
    tomographic reconstruction with uncertain view angles. SIAM/ASA Journal on UQ,
    https://doi.org/10.1137/21M1412268 (2022).

    Parameters
    ----------
    target : `cuqi.distribution.Posterior`
        The target posterior distribution to sample.

    initial_point : ndarray, *Optional*
        Initial parameters.
        If not provided, it defaults to zeros.

    maxit : int
        Maximum number of inner iterations for solver when generating one sample.
        If not provided, it defaults to 50.

    tol : float
        Tolerance for inner solver.
        The inner solvers will stop before maxit if convergence check reaches tol.
        If not provided, it defaults to 1e-4.

    beta : float
        Smoothing parameter for the Gaussian approximation of the Laplace distribution.
        A small value in the range of 1e-7 to 1e-3 is recommended, though values out of this 
        range might give better results in some cases. Generally, a larger beta value makes 
        sampling easier but results in a worse approximation. See details in Section 3.3 of the paper.
        If not provided, it defaults to 1e-5.

    callback : callable, optional
        A function that will be called after each sampling step. It can be useful for monitoring the sampler during sampling.
        The function should take three arguments: the sampler object, the index of the current sampling step, the total number of requested samples. The last two arguments are integers. An example of the callback function signature is: `callback(sampler, sample_index, num_of_samples)`.
    """
    def __init__(self, target=None, initial_point=None, maxit=50, tol=1e-4, beta=1e-5, **kwargs):

        super().__init__(target=target, initial_point=initial_point, **kwargs)

        # Parameters
        self.maxit = maxit
        self.tol = tol
        self.beta = beta
    
    def _initialize(self):
        self._precompute()

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
    def _data(self):
        return self.target.data - self.target.model._shift

    def _precompute(self):

        D = self.prior._diff_op
        n = D.shape[0]

        # Gaussian approximation of LMRF prior as function of x_k
        def Lk_fun(x_k):
            dd =  1/np.sqrt((D @ x_k)**2 + self.beta*np.ones(n))
            W = sp.sparse.diags(dd)
            return W.sqrt() @ D
        self.Lk_fun = Lk_fun

        self._m = len(self._data)
        self._L1 = self.likelihood.distribution.sqrtprec

        # If prior location is scalar, repeat it to match dimensions
        if len(self.prior.location) == 1:
            self._priorloc = np.repeat(self.prior.location, self.dim)
        else:
            self._priorloc = self.prior.location

        # Initial Laplace approx
        self._L2 = Lk_fun(self.initial_point)
        self._L2mu = self._L2@self._priorloc
        self._b_tild = np.hstack([self._L1@self._data, self._L2mu]) 
        
        # Least squares form
        def M(x, flag):
            if flag == 1:
                out1 = self._L1 @ self.model._forward_func_no_shift(x) # Use forward function which excludes shift
                out2 = np.sqrt(1/self.prior.scale)*(self._L2 @ x)
                out  = np.hstack([out1, out2])
            elif flag == 2:
                idx = int(self._m)
                out1 = self.model._adjoint_func_no_shift(self._L1.T@x[:idx])
                out2 = np.sqrt(1/self.prior.scale)*(self._L2.T @ x[idx:])
                out  = out1 + out2                
            return out
        self.M = M

    def step(self):
        # Update Laplace approximation
        self._L2 = self.Lk_fun(self.current_point)
        self._L2mu = self._L2@self._priorloc
        self._b_tild = np.hstack([self._L1@self._data, self._L2mu]) 
    
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

        # Check Affine model
        if not isinstance(self.likelihood.model, cuqi.model.AffineModel):
            raise TypeError("Model needs to be affine or linear")

        # Check Gaussian likelihood
        if not hasattr(self.likelihood.distribution, "sqrtprec"):
            raise TypeError("Distribution in Likelihood must contain a sqrtprec attribute")

        # Check that prior is LMRF
        if not isinstance(self.prior, cuqi.distribution.LMRF):
            raise ValueError('Unadjusted Gaussian Laplace approximation (UGLA) requires LMRF prior')
        
    def _get_default_initial_point(self, dim):
        """ Get the default initial point for the sampler. Defaults to an array of zeros. """
        return np.zeros(dim)
