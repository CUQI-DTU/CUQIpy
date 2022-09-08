import warnings
import numbers
import numpy as np
import numpy.linalg as nplinalg

import scipy as sp
import scipy.stats as sps
import scipy.sparse as spa
import scipy.linalg as splinalg

try:
    import sksparse.cholmod as skchol
    chols = True
except ImportError:
    chols = False

from cuqi import config
from cuqi.geometry import _get_identity_geometries
from cuqi.utilities import force_ndarray, sparse_cholesky
from cuqi.distribution import Distribution

# TODOs:
# Support sparse covariance matrices without cholmod library

class Gaussian(Distribution):
    """
    General Gaussian probability distribution. Generates instance of cuqi.distribution.Gaussian

    Parameters
    ------------
    mean: Mean of distribution. Can be a scalar or 1D numpy array.
    
    cov: Covariance of the distribution. 
    prec: Precision of the distribution.
    sqrtcov: A matrix R, where R.T@R = CovarianceMatrix of the distribution.
    sqrtprec: A matrix R, where R.T@R = PrecisionMatrix of the distribution.  
    (cov, prec, sqrtcov and sqrtprec can be a scalar, 1D numpy array (assumes diagonal matrix), or 2D sparse or numpy arrays).
    
    Note: 2D sparse arrays are handled if the cholmod library is installed, otherwise we get an error.
    
    Example
    -----------
    .. code-block:: python

        # Generate an i.i.d. n-dim Gaussian with zero mean and 2 variance.
        n = 4
        x = cuqi.distribution.Gaussian(mean=np.zeros(n), cov=2)
    """
    def __init__(self, mean=None, cov=None, prec=None, sqrtcov=None, sqrtprec=None, is_symmetric=True, **kwargs):
        super().__init__(is_symmetric=is_symmetric, **kwargs)

        self.mean = mean

        if (cov is not None) + (prec is not None) + (sqrtprec is not None) + (sqrtcov is not None) == 0:
            self._mutable_vars = ['mean', 'cov']
            self.cov = cov
        elif (cov is not None) + (prec is not None) + (sqrtprec is not None) + (sqrtcov is not None) != 1:
            raise ValueError("Exactly one of 'cov', 'prec', 'sqrtcov', or 'sqrtprec' may be specified")
        
        if cov is not None:
            self._mutable_vars = ['mean', 'cov']
            self.cov = cov
        elif prec is not None:
            self._mutable_vars = ['mean', 'prec']
            self.prec = prec
        elif sqrtcov is not None:
            self._mutable_vars = ['mean', 'sqrtcov']
            self.sqrtcov = sqrtcov
        elif sqrtprec is not None:
            self._mutable_vars = ['mean', 'sqrtprec']
            self.sqrtprec = sqrtprec

    @property
    def mean(self):
        return self._mean

    @mean.setter
    def mean(self, value):
        self._mean = force_ndarray(value, flatten=True)

    @property
    def cov(self):
        return self._cov

    @cov.setter
    def cov(self, value):
        if not 'cov' in self._mutable_vars:
            raise ValueError(f"Mutable variables are {self._mutable_vars}")        
        value = force_ndarray(value)
        self._cov = value       
        if (value is not None) and (not callable(value)):  
            if self.dim > config.MIN_DIM_SPARSE:
                dimflag = True # do sparse computations
            else:
                dimflag = False  # use numpy
            prec, sqrtprec, logdet, rank = get_sqrtprec_from_cov(self.dim, value, dimflag)
            self._prec = prec
            self._sqrtprec = sqrtprec
            self._logdet = logdet
            self._rank = rank

    @property
    def prec(self):
        return self._prec

    @prec.setter
    def prec(self, value):
        if not 'prec' in self._mutable_vars:
            raise ValueError(f"Mutable variables are {self._mutable_vars}")        
        value = force_ndarray(value)
        self._prec = value       
        if (value is not None) and (not callable(value)):  
            if self.dim > config.MIN_DIM_SPARSE:
                dimflag = True # do sparse computations
            else:
                dimflag = False  # use numpy
            sqrtprec, logdet, rank = get_sqrtprec_from_prec(self.dim, value, dimflag)
            self._cov = None  # Often not necessary to compute cov
            self._sqrtprec = sqrtprec
            self._logdet = logdet
            self._rank = rank

    @property
    def sqrtcov(self):        
        return self._sqrtcov

    @sqrtcov.setter
    def sqrtcov(self, value):
        if not 'sqrtcov' in self._mutable_vars:
            raise ValueError(f"Mutable variables are {self._mutable_vars}")        
        value = force_ndarray(value)
        self._sqrtcov = value       
        if (value is not None) and (not callable(value)):  
            if self.dim > config.MIN_DIM_SPARSE:
                dimflag = True # do sparse computations
            else:
                dimflag = False  # use numpy
            prec, sqrtprec, logdet, rank = get_sqrtprec_from_sqrtcov(self.dim, value, dimflag)
            self._cov = None  # Often not necessary to compute cov
            self._prec = prec
            self._sqrtprec = sqrtprec
            self._logdet = logdet
            self._rank = rank

    @property
    def sqrtprec(self):        
        return self._sqrtprec

    @sqrtprec.setter
    def sqrtprec(self, value):
        if not 'sqrtprec' in self._mutable_vars:
            raise ValueError(f"Mutable variables are {self._mutable_vars}")        
        value = force_ndarray(value)
        self._sqrtprec = value
        if (value is not None) and (not callable(value)):  
            if self.dim > config.MIN_DIM_SPARSE:
                dimflag = True # do sparse computations
            else:
                dimflag = False  # use numpy
            sqrtprec, logdet, rank = get_sqrtprec_from_sqrtprec(self.dim, value, dimflag)
            self._cov = None  # Often not necessary to compute cov
            self._sqrtprec = sqrtprec 
            self._logdet = logdet
            self._rank = rank

    @property
    def logdet(self):        
        return self._logdet

    @property
    def rank(self):        
        return self._rank

    @property
    def sqrtprecTimesMean(self):
        return (self.sqrtprec@self.mean).flatten()

    def _logupdf(self, x):
        # compute unnormalized density
        dev = x - self.mean
        mahadist = np.sum(np.square(self.sqrtprec @ dev.T), axis=0)
        # dev.T @ self.prec @ dev
        return -0.5*mahadist.flatten()

    def logpdf(self, x):
        if self.logdet is None:
            raise NotImplementedError("Normalized density is not implemented for Gaussian when precision or covariance is sparse and cholmod is not installed.")
        Z = -0.5*(self.rank*np.log(2*np.pi) + self.logdet.flatten())  # normalizing constant
        logup = self._logupdf(x)                            # unnormalized density
        return Z + logup

    def cdf(self, x1):   # no closed form, we rely on scipy
        return sps.multivariate_normal.cdf(x1, self.mean, self.cov)

    def gradient(self, val, *args, **kwargs):
        #Avoid complicated geometries that change the gradient.
        if not type(self.geometry) in _get_identity_geometries() and \
           not hasattr(self.geometry, 'gradient'):
            raise NotImplementedError("Gradient not implemented for distribution {} with geometry {}".format(self,self.geometry))

        if not callable(self.mean): # for prior
            return -( self.prec @ (val - self.mean).T )
        elif hasattr(self.mean, "gradient"): # for likelihood
            model = self.mean
            dev = val - model.forward(*args, **kwargs)
            if isinstance(dev, numbers.Number):
                dev = np.array([dev])
            return model.gradient(self.prec @ dev, *args, **kwargs)
        else:
            warnings.warn('Gradient not implemented for {}'.format(type(self.mean)))

    def _sample(self, N=1, rng=None):
        """ Generate samples of the Gaussian distribution using
        `s = mean + pseudoinverse(sqrtprec)*e`,
        where `e` is a standard Gaussian random vector and `s` is the desired sample 
        """
        # Sample N(0,I)
        if rng is not None:
            e = rng.randn(np.shape(self.sqrtprec)[0], N)
        else:
            e = np.random.randn(np.shape(self.sqrtprec)[0], N)

        # Compute perturbation
        if spa.issparse(self.sqrtprec): # do sparse
            # if np.allclose(self.sqrtprec, spa.tril(self.sqrtprec, format='csr')): # matrix is triangular
            #     perturbation = spa.linalg.spsolve_triangular(self.sqrtprec, e)
            if (N == 1):
                perturbation = spa.linalg.spsolve(self.sqrtprec, e)[:, None]
            else:
                perturbation = spa.linalg.spsolve(self.sqrtprec, e)
        else:
            if np.allclose(self.sqrtprec, np.tril(self.sqrtprec)): # matrix is triangular
                perturbation = splinalg.solve_triangular(self.sqrtprec, e)
            else:
                perturbation = splinalg.solve(self.sqrtprec, e)

        # Add mean
        s = self.mean[:, None] + perturbation
        return s

#===================================================
#===================================================
#===================================================
def get_sqrtprec_from_cov(dim, cov, dimflag):
    # cov is scalar, corrmat is identity or 1D
    if (cov.shape[0] == 1): 
        var = cov.ravel()[0]
        logdet = dim*np.log(var)
        rank = dim
        if dimflag:
            prec = (1/var)*spa.identity(dim, format="csr")
            sqrtprec = np.sqrt(1/var)*spa.identity(dim, format="csr")
        else:
            prec = (1/var)*np.identity(dim)
            sqrtprec = np.sqrt(1/var)*np.identity(dim)

    # cov is vector
    elif not spa.issparse(cov) and cov.shape[0] == np.size(cov): 
        logdet = np.sum(np.log(cov))
        rank = dim
        if dimflag:
            prec = spa.diags(1/cov, format="csr")
            sqrtprec = spa.diags(np.sqrt(1/cov), format="csr")
        else:
            prec = np.diag(1/cov)
            sqrtprec = np.diag(np.sqrt(1/cov))

    # cov diagonal
    #(spa.issparse(cov) and cov.format == 'dia') or (not spa.issparse(cov) and np.count_nonzero(cov-np.diag(np.diagonal(cov))) == 0): 
    elif np.count_nonzero(cov-np.diag(cov.diagonal())) == 0:
        var = cov.diagonal()
        logdet = np.sum(np.log(var))
        rank = dim
        if dimflag:
            prec = spa.diags(1/var, format="csr")
            sqrtprec = spa.diags(np.sqrt(1/var), format="csr")
        else:
            prec = np.diag(1/var)
            sqrtprec = np.diag(np.sqrt(1/var))

    # cov is full
    else:
        if spa.issparse(cov):
            if chols:
                L_cholmod = skchol.cholesky(cov, ordering_method='natural')
                prec = L_cholmod.inv()
                sqrtprec = sparse_cholesky(prec)
                logdet = L_cholmod.logdet()
                rank = spa.csgraph.structural_rank(cov)# or nplinalg.matrix_rank(cov.todense())
                # sqrtcov = L_cholmod.L()
            else:
                #raise NotImplementedError("Sparse covariance is only supported via 'cholmod'.")
                prec = spa.linalg.inv(cov)
                sqrtprec = sparse_cholesky(prec)
                logdet = None # np.log(nplinalg.det(cov.todense())) TODO
                rank = spa.csgraph.structural_rank(cov)                        
        else:
            if not np.allclose(cov, cov.T):
                raise ValueError("Covariance matrix has to be symmetric.") 
            if dimflag:
                # this comes from scipy implementation
                s, u = splinalg.eigh(cov, check_finite=True)
                eps = eigvalsh_to_eps(s)
                if np.min(s) < -eps:
                    raise ValueError("The input matrix must be symmetric positive semidefinite.")                    
                d = s[s > eps]                    
                s_pinv = np.array([0 if abs(x) <= eps else 1/x for x in s], dtype=float)
                
                U = np.multiply(u, np.sqrt(s_pinv))
                sqrtprec = U @ np.diag(np.sign(np.diag(U))) #ensure sign is deterministic (scipy gives non-deterministic result)
                sqrtprec = U.T # We want to have the columns as the eigenvectors
                
                rank = len(d)
                logdet = np.sum(np.log(d))
                prec = sqrtprec.T @ sqrtprec
            else:
                rank = nplinalg.matrix_rank(cov)
                logdet = np.log(nplinalg.det(cov))
                prec = nplinalg.inv(cov)
                sqrtprec = nplinalg.cholesky(prec).T
    return prec, sqrtprec, logdet, rank

#===================================================
def get_sqrtprec_from_prec(dim, prec, dimflag):
    # prec is scalar, corrmat is identity or 1D
    if (prec.shape[0] == 1): 
        precision = prec.ravel()[0]
        logdet = -dim*np.log(precision)
        rank = dim
        if dimflag:
            # cov = (1/precision)*spa.identity(dim, format="csr")
            sqrtprec = np.sqrt(precision)*spa.identity(dim, format="csr")
        else:
            # cov = (1/precision)*np.identity(dim)
            sqrtprec = np.sqrt(precision)*np.identity(dim)

    # prec is vector
    elif not spa.issparse(prec) and prec.shape[0] == np.size(prec): 
        logdet = np.sum(-np.log(prec))
        rank = dim
        if dimflag:
            # cov = spa.diags(1/prec, format="csr")
            sqrtprec = spa.diags(np.sqrt(prec), format="csr")
        else:
            # cov = np.diag(1/cov)
            sqrtprec = np.diag(np.sqrt(prec))

    # prec diagonal
    elif np.count_nonzero(prec-np.diag(prec.diagonal())) == 0:
        precision = prec.diagonal()
        logdet = np.sum(-np.log(precision))
        rank = dim
        if dimflag:
            # cov = spa.diags(1/precision, format="csr")
            sqrtprec = spa.diags(np.sqrt(precision), format="csr")
        else:
            # cov = np.diag(1/precision)
            sqrtprec = np.diag(np.sqrt(precision))

    # prec is full
    else:
        if spa.issparse(prec):
            if chols:
                L_cholmod = skchol.cholesky(prec, ordering_method='natural')
                sqrtprec = L_cholmod.L()
                # cov = L_cholmod.inv()
                logdet = -L_cholmod.logdet()
                rank = spa.csgraph.structural_rank(prec)# or nplinalg.matrix_rank(cov.todense())
            else:
                # cov = spa.linalg.inv(cov)
                sqrtprec = sparse_cholesky(prec)
                logdet = None # np.log(nplinalg.det(cov.todense()))
                rank = spa.csgraph.structural_rank(prec)                    
        else:
            if not np.allclose(prec, prec.T):
                raise ValueError("Precision matrix has to be symmetric.") 
            if dimflag:
                s, u = splinalg.eigh(prec, check_finite=True)
                eps = eigvalsh_to_eps(s)
                if np.min(s) < -eps:
                    raise ValueError("The input matrix must be symmetric positive semidefinite.")                    
                d = s[s > eps]
                
                U = np.multiply(u, np.sqrt(s))
                sqrtprec = U @ np.diag(np.sign(np.diag(U))) #ensure sign is deterministic (scipy gives non-deterministic result)
                sqrtprec = U.T # We want to have the columns as the eigenvectors
                
                rank = len(d)
                logdet = -np.sum(np.log(d))
            else:
                rank = nplinalg.matrix_rank(prec)
                logdet = -np.log(nplinalg.det(prec))
                # cov = nplinalg.inv(prec)
                sqrtprec = nplinalg.cholesky(prec).T
    return sqrtprec, logdet, rank

#===================================================
#===================================================
#===================================================
def get_sqrtprec_from_sqrtcov(dim, sqrtcov, dimflag):
    # sqrtcov is scalar, corrmat is identity or 1D
    if (sqrtcov.shape[0] == 1): 
        sqrtcov = sqrtcov.ravel()[0]
        var = sqrtcov**2
        logdet = dim*np.log(var)
        rank = dim
        if dimflag:
            prec = (1/var)*spa.identity(dim, format="csr")
            sqrtprec = (1/sqrtcov)*spa.identity(dim, format="csr")
        else:
            prec = (1/var)*np.identity(dim)
            sqrtprec = (1/sqrtcov)*np.identity(dim)

    # sqrtcov is vector
    elif not spa.issparse(sqrtcov) and sqrtcov.shape[0] == np.size(sqrtcov): 
        cov = sqrtcov**2
        logdet = np.sum(np.log(cov))
        rank = dim
        if dimflag:
            prec = spa.diags(1/cov, format="csr")
            sqrtprec = spa.diags(1/sqrtcov, format="csr")
        else:
            prec = np.diag(1/cov)
            sqrtprec = np.diag(1/sqrtcov)

    # sqrtcov diagonal
    elif np.count_nonzero(sqrtcov-np.diag(sqrtcov.diagonal())) == 0: 
        std = sqrtcov.diagonal()
        var = std**2
        logdet = np.sum(np.log(var))
        rank = dim
        if dimflag:
            prec = spa.diags(1/var, format="csr")
            sqrtprec = spa.diags(1/std, format="csr")
        else:
            prec = np.diag(1/var)
            sqrtprec = np.diag(1/std)

    # sqrtcov is full
    else:
        if spa.issparse(sqrtcov):
            if chols:
                cov = sqrtcov@sqrtcov.T
                L_cholmod = skchol.cholesky(cov, ordering_method='natural')
                prec = L_cholmod.inv()
                sqrtprec = sparse_cholesky(prec)
                logdet = L_cholmod.logdet()
                rank = spa.csgraph.structural_rank(cov)# or nplinalg.matrix_rank(cov.todense())
                # sqrtcov = L_cholmod.L()
            else:
                raise NotImplementedError("Sparse standard deviation is only supported via 'cholmod'.")
                # TODO:
                # prec = spa.linalg.inv(cov)
                # sqrtprec = sparse_cholesky(prec)
                # logdet = np.log(nplinalg.det(cov.todense()))
                # rank = spa.csgraph.structural_rank(cov)                    
        else:
            if dimflag:
                # this comes from scipy implementation
                cov = sqrtcov@sqrtcov.T
                s, u = splinalg.eigh(cov, check_finite=True)
                eps = eigvalsh_to_eps(s)
                if np.min(s) < -eps:
                    raise ValueError("The input matrix must be symmetric positive semidefinite.")                    
                d = s[s > eps]                    
                s_pinv = np.array([0 if abs(x) <= eps else 1/x for x in s], dtype=float)
                
                U = np.multiply(u, np.sqrt(s_pinv))
                sqrtprec = U @ np.diag(np.sign(np.diag(U))) #ensure sign is deterministic (scipy gives non-deterministic result)
                sqrtprec = U.T # We want to have the columns as the eigenvectors
                
                rank = len(d)
                logdet = np.sum(np.log(d))
                prec = sqrtprec.T @ sqrtprec
            else:
                cov = sqrtcov@sqrtcov.T
                rank = nplinalg.matrix_rank(cov)
                logdet = np.log(nplinalg.det(cov))
                prec = nplinalg.inv(cov)
                sqrtprec = nplinalg.cholesky(prec).T
    return prec, sqrtprec, logdet, rank

#===================================================
def get_sqrtprec_from_sqrtprec(dim, sqrtprec, dimflag):
    # sqrtprec is scalar, corrmat is identity or 1D
    if (sqrtprec.shape[0] == 1): 
        logdet = -dim*np.log(sqrtprec**2)
        rank = dim
        dia = np.ones(dim)*sqrtprec.flatten()
        if dimflag:
            sqrtprec = spa.diags(dia)
        else:
            sqrtprec = np.diag(dia)
    # sqrtprec is vector
    elif not spa.issparse(sqrtprec) and sqrtprec.shape[0] == np.size(sqrtprec): 
        logdet = np.sum(-np.log(sqrtprec**2))
        rank = dim
        if dimflag:
            sqrtprec = spa.diags(sqrtprec)
        else:
            sqrtprec = np.diag(sqrtprec)
    # sqrtprec diagonal
    elif np.count_nonzero(sqrtprec-np.diag(sqrtprec.diagonal())) == 0:
        stdinv = sqrtprec.diagonal()
        precision = stdinv**2
        logdet = np.sum(-np.log(precision))
        rank = dim
    # sqrtprec is full
    else:
        if spa.issparse(sqrtprec):
            if chols:
                prec = sqrtprec@sqrtprec.T
                L_cholmod = skchol.cholesky(prec, ordering_method='natural')
                logdet = -L_cholmod.logdet()
                rank = spa.csgraph.structural_rank(prec)# or nplinalg.matrix_rank(cov.todense())
            else:
                raise NotImplementedError("Sparse precision is only supported via 'cholmod'.")
                # TODO:               
        else:
            if dimflag:
                prec = sqrtprec@sqrtprec.T
                s, _ = splinalg.eigh(prec, check_finite=True)
                eps = eigvalsh_to_eps(s)
                if np.min(s) < -eps:
                    raise ValueError("The input matrix must be symmetric positive semidefinite.")                    
                d = s[s > eps]
                rank = len(d)
                logdet = -np.sum(np.log(d))
            else:
                prec = sqrtprec@sqrtprec.T
                rank = nplinalg.matrix_rank(prec)
                logdet = -np.log(nplinalg.det(prec))
    return sqrtprec, logdet, rank

#===================================================
def eigvalsh_to_eps(spectrum, cond=None, rcond=None):
    #Determine which eigenvalues are "small" given the spectrum.
    if rcond is not None:
        cond = rcond
    if cond in [None, -1]:
        t = spectrum.dtype.char.lower()
        factor = {'f': 1E3, 'd': 1E6}
        cond = factor[t] * np.finfo(t).eps
    eps = cond * np.max(abs(spectrum))
    return eps

class JointGaussianSqrtPrec(Distribution):
    """
    Joint Gaussian probability distribution defined by means and sqrt of precision matricies of independent Gaussians.
    Generates instance of cuqi.distribution.JointGaussianSqrtPrec.

    
    Parameters
    ------------
    means: List of means for each Gaussian distribution.
    sqrtprecs: List of sqrt precision matricies for each Gaussian distribution.

    Attributes
    ------------
    sqrtprec: Returns the sqrt precision matrix of the joined gaussian in stacked form.
    sqrtprecTimesMean: Returns the sqrt precision matrix times the mean of the distribution.
    
    Methods
    -----------
    sample: generate one or more random samples (NotImplemented)
    pdf: evaluate probability density function (NotImplemented)
    logpdf: evaluate log probability density function (NotImplemented)
    cdf: evaluate cumulative probability function (NotImplemented)
    """    
    def __init__(self,means=None,sqrtprecs=None,is_symmetric=True,**kwargs):

        # Check if given as list
        if not isinstance(means,list) or not isinstance(sqrtprecs,list):
            raise ValueError("Means and sqrtprecs need to be a list of vectors and matrices respectively.")

        # Force to numpy arrays
        for i in range(len(means)):
            means[i] = force_ndarray(means[i],flatten=True)
        for i in range(len(sqrtprecs)):
            sqrtprecs[i] = force_ndarray(sqrtprecs[i])

        # Check dimension match TODO: move to setter methods for means and sqrtprecs
        dim1 = len(means[0])
        for mean in means:
            if dim1 != len(mean):
                raise ValueError("All means must have the same dimension")
        dim2 = sqrtprecs[0].shape[1]
        for sqrtprec in sqrtprecs:
            if dim2 != sqrtprec.shape[1]:
                raise ValueError("All sqrtprecs must have the same number of columns")

        super().__init__(is_symmetric=is_symmetric,**kwargs)

        self._means = means
        self._sqrtprecs = sqrtprecs
        self._dim = max(dim1,dim2)

    def _sample(self,N):
        raise NotImplementedError("Sampling not implemented")

    def logpdf(self,x):
        raise NotImplementedError("pdf not implemented")

    @property
    def dim(self):
        return self._dim

    @property
    def sqrtprec(self):
        if spa.issparse(self._sqrtprecs[0]):
            return spa.vstack((self._sqrtprecs))
        else:
            return np.vstack((self._sqrtprecs))

    @property
    def sqrtprecTimesMean(self):
        result = []
        for i in range(len(self._means)):
            result.append((self._sqrtprecs[i]@self._means[i]).flatten())
        return np.hstack(result)
