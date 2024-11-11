import scipy as sp
import numpy as np
import cuqi
from cuqi.distribution import Normal
from cuqi.solver import CGLS
from cuqi.sampler import Sampler


class UGLA(Sampler):
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

    x0 : ndarray
        Initial parameters. *Optional*

    maxit : int
        Maximum number of inner iterations for solver when generating one sample.

    tol : float
        Tolerance for inner solver. Will stop before maxit if the inner solvers convergence check reaches tol.

    beta : float
        Smoothing parameter for the Gaussian approximation of the Laplace distribution. Larger beta is easier to sample but is a worse approximation.

    rng : np.random.RandomState
        Random number generator used for sampling. *Optional*

    callback : callable, *Optional*
        If set this function will be called after every sample.
        The signature of the callback function is `callback(sample, sample_index)`,
        where `sample` is the current sample and `sample_index` is the index of the sample.
        An example is shown in demos/demo31_callback.py.

    Returns
    -------
    cuqi.samples.Samples
        Samples from the posterior distribution.

    """

    def __init__(self, target, x0=None, maxit=50, tol=1e-4, beta=1e-5, rng=None, **kwargs):
        
        super().__init__(target, x0=x0, **kwargs)

        # Check target type
        if not isinstance(self.target, cuqi.distribution.Posterior):
            raise ValueError(f"To initialize an object of type {self.__class__}, 'target' need to be of type 'cuqi.distribution.Posterior'.")       

        # Check Affine model
        if not isinstance(self.target.likelihood.model, cuqi.model.AffineModel):
            raise TypeError("Model needs to be affine or linear")

        # Check Gaussian likelihood
        if not hasattr(self.target.likelihood.distribution, "sqrtprec"):
            raise TypeError("Distribution in Likelihood must contain a sqrtprec attribute")

        # Check that prior is LMRF
        if not isinstance(self.target.prior, cuqi.distribution.LMRF):
            raise ValueError('Unadjusted Gaussian Laplace approximation (UGLA) requires LMRF prior')

        # Modify initial guess since Sampler sets it to ones.       
        if x0 is not None:
            self.x0 = x0
        else:
            self.x0 = np.zeros(self.target.prior.dim)
        
        # Store internal parameters
        self.maxit = maxit
        self.tol = tol
        self.beta = beta
        self.rng = rng

    def _sample_adapt(self, Ns, Nb):
        return self._sample(Ns, Nb)

    def _sample(self, Ns, Nb):
        """ Sample from the approximate posterior.

        Parameters
        ----------
        Ns : int
            Number of samples to draw.

        Nb : int
            Number of burn-in samples to discard.

        Returns
        -------
        samples : ndarray
            Samples from the approximate posterior.

        target_eval : ndarray
            Log-likelihood of each sample.

        acc : ndarray
            Acceptance rate of each sample.

        """

        # Extract diff_op from target prior
        D = self.target.prior._diff_op
        n = D.shape[0]

        # Gaussian approximation of LMRF prior as function of x_k
        def Lk_fun(x_k):
            dd =  1/np.sqrt((D @ x_k)**2 + self.beta*np.ones(n))
            W = sp.sparse.diags(dd)
            return W.sqrt() @ D

        # Now prepare "LinearRTO" type sampler. TODO: Use LinearRTO for this instead
        self._shift = 0

        # Pre-computations
        self._model = self.target.likelihood.model   
        self._data = self.target.likelihood.data - self.target.model._shift
        self._m = len(self._data)
        self._L1 = self.target.likelihood.distribution.sqrtprec

        # If prior location is scalar, repeat it to match dimensions
        if len(self.target.prior.location) == 1:
            self._priorloc = np.repeat(self.target.prior.location, self.dim)
        else:
            self._priorloc = self.target.prior.location

        # Initial Laplace approx
        self._L2 = Lk_fun(self.x0)
        self._L2mu = self._L2@self._priorloc
        self._b_tild = np.hstack([self._L1@self._data, self._L2mu]) 
        
        #self.n = len(self.x0)
        
        # Least squares form
        def M(x, flag):
            if flag == 1:
                out1 = self._L1 @ self._model._forward_func_no_shift(x) # Use forward function which excludes shift
                out2 = np.sqrt(1/self.target.prior.scale)*(self._L2 @ x)
                out  = np.hstack([out1, out2])
            elif flag == 2:
                idx = int(self._m)
                out1 = self._model._adjoint_func_no_shift(self._L1.T@x[:idx])
                out2 = np.sqrt(1/self.target.prior.scale)*(self._L2.T @ x[idx:])
                out  = out1 + out2                
            return out 
        
        # Initialize samples
        N = Ns+Nb   # number of simulations        
        samples = np.empty((self.target.dim, N))
                     
        # initial state   
        samples[:, 0] = self.x0
        for s in range(N-1):

            # Update Laplace approximation
            self._L2 = Lk_fun(samples[:, s])
            self._L2mu = self._L2@self._priorloc
            self._b_tild = np.hstack([self._L1@self._data, self._L2mu]) 
        
            # Sample from approximate posterior
            e = Normal(mean=np.zeros(len(self._b_tild)), std=1).sample(rng=self.rng)
            y = self._b_tild + e # Perturb data
            sim = CGLS(M, y, samples[:, s], self.maxit, self.tol, self._shift)            
            samples[:, s+1], _ = sim.solve()

            self._print_progress(s+2,N) #s+2 is the sample number, s+1 is index assuming x0 is the first sample
            self._call_callback(samples[:, s+1], s+1)

        # remove burn-in
        samples = samples[:, Nb:]
        
        return samples, None, None
