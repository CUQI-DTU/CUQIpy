import numpy as np
import scipy.stats as sps
from scipy.sparse import diags, identity, issparse, vstack, csc_matrix, isspmatrix_csr, isspmatrix_csc
from scipy.sparse import linalg as splinalg
from scipy.linalg import eigh, eigvals, cholesky
from cuqi.geometry import _get_identity_geometries
from cuqi.utilities import force_ndarray, sparse_cholesky
from cuqi import config
import warnings
import numbers
from cuqi.distribution import Distribution

class GaussianCov(Distribution): # TODO: super general with precisions
    """
    General Gaussian probability distribution. Generates instance of cuqi.distribution.GaussianCov

    
    Parameters
    ------------
    mean: Mean of distribution. Can be a scalar or 1d numpy array
    cov: Covariance of distribution. Can be a scalar, 1d numpy array (assumes diagonal elements), or 2d numpy array.
    
    Example
    -----------
    .. code-block:: python

        # Generate an i.i.d. n-dim Gaussian with zero mean and 2 variance.
        n = 4
        x = cuqi.distribution.GaussianCov(mean=np.zeros(n), cov=2)
    """
    def __init__(self, mean=None, cov=None, is_symmetric=True, **kwargs):
        super().__init__(is_symmetric=is_symmetric, **kwargs) 

        self.mean = mean
        self.cov = cov

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
        value = force_ndarray(value)
        self._cov = value
        if (value is not None) and (not callable(value)):
            prec, sqrtprec, logdet, rank = self.get_prec_from_cov(value)
            self._prec = prec
            self._sqrtprec = sqrtprec
            self._logdet = logdet
            self._rank = rank

    @property
    def sqrtprec(self):        
        return self._sqrtprec
    @property
    def prec(self):        
        return self._prec
    @property
    def logdet(self):        
        return self._logdet
    @property
    def rank(self):        
        return self._rank

    @property
    def sqrtprecTimesMean(self):
        return (self.sqrtprec@self.mean).flatten()

    @property 
    def Sigma(self): #Backwards compatabilty. TODO. Remove Sigma in demos, tests etc.
        if self.dim > config.MAX_DIM_INV:
            raise NotImplementedError(f"Sigma: Full covariance matrix not implemented for dim > {config.MAX_DIM_INV}.")
        return np.linalg.inv(self.prec.toarray())  

    def get_prec_from_cov(self, cov, eps = 1e-5):
        # if cov is scalar, corrmat is identity or 1D
        if (cov.shape[0] == 1): 
            var = cov.ravel()[0]
            prec = (1/var)*identity(self.dim)
            sqrtprec = np.sqrt(1/var)*identity(self.dim)
            logdet = self.dim*np.log(var)
            rank = self.dim
        # Cov is vector
        elif not issparse(cov) and cov.shape[0] == np.size(cov): 
            prec = diags(1/cov)
            sqrtprec = diags(np.sqrt(1/cov))
            logdet = np.sum(np.log(cov))
            rank = self.dim
        # Cov diagonal
        elif (issparse(cov) and cov.format == 'dia') or (not issparse(cov) and np.count_nonzero(cov-np.diag(np.diagonal(cov))) == 0): 
            var = cov.diagonal()
            prec = diags(1/var)
            sqrtprec = diags(np.sqrt(1/var))
            logdet = np.sum(np.log(var))
            rank = self.dim
        # Cov is full
        else:
            if issparse(cov):
                raise NotImplementedError("Sparse covariance is not supported for now")
                #from sksparse.cholmod import cholesky #Uses package sksparse>=0.1
                #cholmodcov = None #cholesky(cov, ordering_method='natural')
                #sqrtcov = cholmodcov.L()
                #logdet = cholmodcov.logdet()
            else:
                # Can we use cholesky factorization and somehow get the logdet also?
                s, u = eigh(cov, lower=True, check_finite=True)
                d = s[s > eps]
                s_pinv = np.array([0 if abs(x) <= eps else 1/x for x in s], dtype=float)
                sqrtprec = np.multiply(u, np.sqrt(s_pinv)) 
                sqrtprec = sqrtprec@diags(np.sign(np.diag(sqrtprec))) #ensure sign is deterministic (scipy gives non-deterministic result)
                sqrtprec = sqrtprec.T # We want to have the columns as the eigenvectors
                rank = len(d)
                logdet = np.sum(np.log(d))
                prec = sqrtprec.T @ sqrtprec

        return prec, sqrtprec, logdet, rank     

    def logpdf(self, x):
        # organize shape of inputs
        # x = x.reshape(-1, self.dim)

        # compute density
        dev = x - self.mean
        mahadist = np.sum(np.square(self.sqrtprec @ dev.T), axis=0)
        return -0.5*(self.rank*np.log(2*np.pi) + self.logdet + mahadist)

    def cdf(self, x1):   # TODO
        return sps.multivariate_normal.cdf(x1, self.mean, self.cov)

    def gradient(self, val, *args, **kwargs):
        # organize shape of inputs
        # val = val.reshape(-1, self.dim)

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
        # If scalar or vector cov use numpy normal
        if (self.cov.shape[0] == 1) or (not issparse(self.cov) and self.cov.shape[0] == np.size(self.cov)): 
            return self._sample_using_sqrtprec(N, rng)  

        elif issparse(self.cov) and issparse(self.sqrtprec):        
            return self._sample_using_sqrtprec(N, rng)

        else:
            return self._sample_using_sqrtprec(N, rng)
     
    def _sample_using_sqrtprec(self, N=1, rng=None):
        """ Generate samples of the Gaussian distribution using
        `s = mean + pseudoinverse(sqrtprec)*eps`,
        where `eps` is a standard normal noise and `s`is the desired sample 
        """
        #Sample N(0,I)
        if rng is not None:
            e = rng.randn(np.shape(self.sqrtprec)[0],N)
        else:
            e = np.random.randn(np.shape(self.sqrtprec)[0],N)

        # Convert matrix to sparse (this avoids the scipy warning)
        # TODO. REMOVE THIS IN NEW GAUSSIAN (including imports)
        sqrtprec = self.sqrtprec
        if not (isspmatrix_csc(sqrtprec) or isspmatrix_csr(sqrtprec)):
            sqrtprec = csc_matrix(sqrtprec)

        #Compute permutation
        if N==1: #Ensures we add (dim,1) with (dim,1) and not with (dim,)
            permutation = splinalg.spsolve(sqrtprec,e)[:,None]
        else:
            permutation = splinalg.spsolve(sqrtprec,e)
            
        # Add to mean
        s = self.mean[:,None] + permutation
        return s

class GaussianSqrtPrec(Distribution):
    """
    Gaussian probability distribution defined using sqrt of precision matrix. 
    Generates instance of cuqi.distribution.GaussianSqrtPrec
    
    Parameters
    ------------
    mean: Mean of distribution. 1d numpy array
    sqrtprec: A matrix R, where R.T@R = PrecisionMatrix of the distribution. Can be a 2d sparse or numpy array.
    
    Methods
    -----------
    sample: generate one or more random samples
    pdf: evaluate probability density function
    logpdf: evaluate log probability density function
    
    Example
    -----------
    # Generate an i.i.d. n-dim Gaussian with zero mean and some standard deviation std.
    x = cuqi.distribution.Normal(mean=np.zeros(n), sqrtprec = 1/std*np.eye)
    """
    def __init__(self, mean=None, sqrtprec=None, is_symmetric=True, **kwargs):
        # Init from abstract distribution class
        super().__init__(is_symmetric=is_symmetric, **kwargs)
        self.mean = force_ndarray(mean, flatten=True)
        self.sqrtprec = force_ndarray(sqrtprec)

    def _sample(self, N):
        
        if N == 1:
            mean = self.mean
        else:
            mean = self.mean[:,None]

        samples = mean + splinalg.spsolve(self.sqrtprec, np.random.randn(np.shape(self.sqrtprec)[0], N))
        #samples = mean + lstsq(self.sqrtprec, np.random.randn(np.shape(self.sqrtprec)[0], N), cond = 1e-14)[0] #Non-sparse
        
        return samples

    def logpdf(self, x):
        # sqrtprec is scalar
        if (self.sqrtprec.shape[0] == 1): 
            prec = ((self.sqrtprec[0][0]**2)*identity(self.dim)).diagonal()
            sqrtprec = np.sqrt(prec)
            logdet = np.sum(np.log(prec))
            rank = self.dim
        # sqrtprec is vector
        elif not issparse(self.sqrtprec) and self.sqrtprec.shape[0] == np.size(self.sqrtprec): 
            prec = diags(self.sqrtprec**2)
            sqrtprec = diags(self.sqrtprec)
            logdet = np.sum(np.log(self.sqrtprec**2))
            rank = self.dim
        # Sqrtprec diagonal
        elif (issparse(self.sqrtprec) and self.sqrtprec.format == 'dia'): 
            sqrtprec = self.sqrtprec.diagonal()
            prec =sqrtprec**2
            logdet = np.sum(np.log(prec))
            rank = self.dim
        # Cov is full
        else:
            if issparse(self.sqrtprec):
                raise NotImplementedError("Non-diagonal sparse sqrtprec is not supported for now")
                #from sksparse.cholmod import cholesky #Uses package sksparse>=0.1
                #cholmodcov = None #cholesky(cov, ordering_method='natural')
                #sqrtcov = cholmodcov.L()
                #logdet = cholmodcov.logdet()
            else:
                # Can we use cholesky factorization and somehow get the logdet also?
                eps = np.finfo(float).eps
                s = eigvals(self.sqrtprec.T@self.sqrtprec)
                d = s[s > eps]
                rank = len(d)
                logdet = np.sum(np.log(d))
                sqrtprec = self.sqrtprec

        dev = x - self.mean
        mahadist = np.sum(np.square(sqrtprec @ dev), axis=0)
        # rank(prec) = rank(sqrtprec.T*sqrtprec) = rank(sqrtprec)
        # logdet can also be pseudo-determinant, defined as the product of non-zero eigenvalues
        return -0.5*(rank*np.log(2*np.pi) - logdet + mahadist)

    @property
    def sqrtprecTimesMean(self):
        return (self.sqrtprec@self.mean).flatten()

class GaussianPrec(Distribution):

    def __init__(self, mean, prec, is_symmetric=True, **kwargs):
        super().__init__(is_symmetric=is_symmetric, **kwargs) 

        self.mean = mean
        self.prec = prec

    @property
    def mean(self):
        return self._mean
    
    @mean.setter
    def mean(self, mean):
        self._mean = force_ndarray(mean,flatten=True)

    @property
    def prec(self):
        return self._prec

    @prec.setter
    def prec(self, prec):
        self._prec = force_ndarray(prec)
        # Compute cholesky factorization of precision
        if (prec is not None) and (not callable(prec)):
            if issparse(self._prec):
                self._sqrtprec = sparse_cholesky(self._prec)
                self._rank = self.dim
                self._logdet = 2*sum(np.log(self._sqrtprec.diagonal()))
            else:
                self._sqrtprec = cholesky(self._prec)
                self._rank = self.dim
                self._logdet = 2*sum(np.log(np.diag(self._sqrtprec)))

    def _sample(self, N):
        
        if N == 1:
            mean = self.mean
        else:
            mean = self.mean[:,None]

        samples = mean + splinalg.spsolve(self.sqrtprec, np.random.randn(np.shape(self.sqrtprec)[0], N))
        #samples = mean + lstsq(self.sqrtprec, np.random.randn(np.shape(self.sqrtprec)[0], N), cond = 1e-14)[0] #Non-sparse
        
        return samples

    def logpdf(self, x):
        dev = x - self.mean
        mahadist = np.sum(np.square(self.sqrtprec @ dev), axis=0)
        return -0.5*(self._rank*np.log(2*np.pi) - self._logdet + mahadist)

    def gradient(self, val, *args, **kwargs):
        #Avoid complicated geometries that change the gradient.
        if not type(self.geometry) in _get_identity_geometries():
            raise NotImplementedError("Gradient not implemented for distribution {} with geometry {}".format(self,self.geometry))

        if not callable(self.mean): # for prior
            return -( self.prec @ (val - self.mean) )
        elif hasattr(self.mean,"gradient"): # for likelihood
            model = self.mean
            dev = val - model.forward(*args, **kwargs)
            if isinstance(dev, numbers.Number):
                dev = np.array([dev])
            return model.gradient(self.prec @ dev, *args, **kwargs)
        else:
            warnings.warn('Gradient not implemented for {}'.format(type(self.mean)))

    @property
    def sqrtprec(self):
        return self._sqrtprec

    @property
    def sqrtprecTimesMean(self):
        return (self.sqrtprec@self.mean).flatten()

class Gaussian(GaussianCov):
    """
    Wrapper for GaussianCov using std and corrmat. See ::class::cuqi.distribution.GaussianCov.
    
    Mutable attributes: mean, cov.
    """
    def __init__(self, mean=None, std=None, corrmat=None, is_symmetric=True, **kwargs):
        
        dim = len(mean)
        if dim > config.MAX_DIM_INV:
            raise NotImplementedError("Use GaussianCov for large-scale problems.")
            
        #Compute cov from pre-computations below.
        if corrmat is None:
            corrmat = np.eye(len(mean))
        dim = len(np.diag(corrmat))
        if isinstance(std, (list, tuple, np.ndarray)):
            cov = np.diag(std) @ (corrmat @ np.diag(std))   # covariance
        else:
            cov = np.diag(std*np.ones(dim)) @ (corrmat @ np.diag(std*np.ones(dim)))   # covariance
        super().__init__(mean=mean, cov=cov, is_symmetric=is_symmetric, **kwargs)

    @property 
    def Sigma(self): #Backwards compatabilty. TODO. Remove Sigma in demos, tests etc.
        return self.cov

class JointGaussianSqrtPrec(Distribution):
    """
    Joint Gaussian probability distribution defined by means and sqrt of precision matricies of independent Gaussians.
    Generates instance of cuqi.distribution.JoinedGaussianSqrtPrec.

    
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
        if issparse(self._sqrtprecs[0]):
            return vstack((self._sqrtprecs))
        else:
            return np.vstack((self._sqrtprecs))

    @property
    def sqrtprecTimesMean(self):
        result = []
        for i in range(len(self._means)):
            result.append((self._sqrtprecs[i]@self._means[i]).flatten())
        return np.hstack(result)
            
