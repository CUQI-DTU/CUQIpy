import numpy as np
import scipy.stats as sps
from scipy.special import erf, loggamma, gammainc
from scipy.sparse import diags, spdiags, eye, kron, vstack, identity, issparse
from scipy.sparse import linalg as splinalg
from scipy.linalg import eigh, dft, eigvalsh, pinvh
from cuqi.samples import Samples
from cuqi.geometry import _DefaultGeometry, Geometry
from cuqi.model import LinearModel
from cuqi.utilities import force_ndarray
import warnings

from abc import ABC, abstractmethod
from copy import copy

import inspect

import time

# import sksparse
# from sksparse.cholmod import cholesky
eps = np.finfo(float).eps


# ========== Abstract distribution class ===========
class Distribution(ABC):

    def __init__(self,name=None, geometry=None):
        if not isinstance(name,str) and name is not None:
            raise ValueError("Name must be a string or None")
        self.name = name
        self.is_symmetric = None
        self.geometry = geometry

    @property
    def geometry(self):
        return self._geometry

    @geometry.setter
    def geometry(self,value):
        if isinstance(value, int) or value is None:
            self._geometry = _DefaultGeometry(grid=value)
        elif isinstance(value, Geometry):
            self._geometry = value
        else:
            raise TypeError("The attribute 'geometry' should be of type 'int' or 'cuqi.geometry.Geometry', or None.")

    @abstractmethod
    def logpdf(self,x):
        pass

    def sample(self,N=1,*args,**kwargs):
        #Make sure all values are specified, if not give error
        for key, value in vars(self).items():
            if isinstance(value,Distribution):
                raise NotImplementedError("Parameter {} is {}. Parameter must be a fixed value.".format(key,value))

        # Get samples from the distribution sample method
        s = self._sample(N,*args,**kwargs)

        #Store samples in cuqi samples object if more than 1 sample
        if N==1:
            if len(s) == 1 and isinstance(s,np.ndarray): #Extract single value from numpy array
                s = s.ravel()[0]
            else:
                s = s.flatten()
        else:
            s = Samples(s, self.geometry)

        return s

    @abstractmethod
    def _sample(self,N):
        pass

    def pdf(self,x):
        return np.exp(self.logpdf(x))

    def __call__(self,**kwargs):
        """ Generate new distribution with new attributes given in by keyword arguments """

        # KEYWORD ERROR CHECK
        for kw_key, kw_val in kwargs.items():
            val_found = 0
            for attr_key, attr_val in vars(self).items():
                if kw_key is attr_key:
                    val_found = 1
                elif callable(attr_val) and kw_key in inspect.getfullargspec(attr_val)[0]:
                    val_found = 1
            if val_found == 0:
                raise ValueError("The keyword {} is not part of any attribute or argument to any function of this distribution.".format(kw_key))


        # EVALUATE CONDITIONAL DISTRIBUTION
        new_dist = copy(self) #New cuqi distribution conditioned on the kwargs
        new_dist.name = None  #Reset name to None

        # Go through every attribute and assign values from kwargs accordingly
        for attr_key, attr_val in vars(self).items():
            
            #If keyword directly specifies new value of attribute we simply reassign
            if attr_key in kwargs:
                setattr(new_dist,attr_key,kwargs.get(attr_key))

            #If attribute is callable we check if any keyword arguments can be used as arguments
            if callable(attr_val):

                accepted_keywords = inspect.getfullargspec(attr_val)[0]

                # Builds dict with arguments to call attribute with
                attr_args = {}
                for kw_key, kw_val in kwargs.items():
                    if kw_key in accepted_keywords:
                        attr_args[kw_key] = kw_val

                # If any keywords matched call with those and store output in the new dist
                if len(attr_args)>0:
                    setattr(new_dist,attr_key,attr_val(**attr_args))

        return new_dist

# ========================================================================
class Cauchy_diff(object):

    def __init__(self, location, scale, bndcond):
        self.loc = location
        self.scale = scale
        self.bnd = bndcond

        # finite difference matrix
        one_vec = np.ones(self.dim)
        diags = np.vstack([-one_vec, one_vec])
        if (bndcond == 'zero'):
            locs = [-1, 0]
            Dmat = spdiags(diags, locs, self.dim+1, self.dim)
        elif (bndcond == 'periodic'):
            locs = [-1, 0]
            Dmat = spdiags(diags, locs, self.dim+1, self.dim).tocsr()
            Dmat[-1, 0] = 1
            Dmat[0, -1] = -1
        elif (bndcond == 'neumann'):
            locs = [0, 1]
            Dmat = spdiags(diags, locs, self.dim-1, self.dim)
        elif (bndcond == 'backward'):
            locs = [0, -1]
            Dmat = spdiags(diags, locs, self.dim, self.dim).tocsr()
            Dmat[0, 0] = 1
        elif (bndcond == 'none'):
            Dmat = eye(self.dim)
        self.D = Dmat

    @property
    def dim(self):
        #TODO: handle the case when self.loc = None because len(None) = 1
        return len(self.loc)

    def pdf(self, x):
        Dx = self.D @ (x-self.loc)
        return (1/(np.pi**len(Dx))) * np.prod(self.scale/(Dx**2 + self.scale**2))

    def logpdf(self, x):
        Dx = self.D @ (x-self.loc)
        # g_logpr = (-2*Dx/(Dx**2 + gamma**2)) @ D
        return -len(Dx)*np.log(np.pi) + sum(np.log(self.scale) - np.log(Dx**2 + self.scale**2))
    
    def gradient(self, val, **kwargs):
        if not callable(self.loc): # for prior
            diff = self.D @ val
            return (-2*diff/(diff**2+self.scale**2)) @ self.D
        else:
            warnings.warn('Gradient not implemented for {}'.format(type(self.loc)))
    # def cdf(self, x):   # TODO
    #     return 1/np.pi * np.atan((x-self.loc)/self.scale)

    # def sample(self):   # TODO
    #     return self.loc + self.scale*np.tan(np.pi*(np.random.rand(self.dim)-1/2))


# ========================================================================
class Normal(Distribution):
    """
    Normal probability distribution. Generates instance of cuqi.distribution.Normal

    
    Parameters
    ------------
    mean: mean of distribution
    std: standard deviation
    
    Methods
    -----------
    sample: generate one or more random samples
    pdf: evaluate probability density function
    logpdf: evaluate log probability density function
    cdf: evaluate cumulatiuve probability function
    
    Example
    -----------
    #Generate Normal with mean 2 and standard deviation 1
    p = cuqi.distribution.Normal(mean=2, std=1)
    """
    def __init__(self, mean=None, std=None, **kwargs):
        # Init from abstract distribution class
        if "geometry" not in kwargs.keys() or kwargs["geometry"] is None:
            #TODO: handle the case when self.mean or self.std = None because len(None) = 1
            kwargs["geometry"] = max(np.size(mean),np.size(std))
        super().__init__(**kwargs)  
        self.is_symmetric = True 

        # Init specific to this distribution
        self.mean = mean
        self.std = std


    @property
    def dim(self):
        return self.geometry.dim

    def pdf(self, x):
        return 1/(self.std*np.sqrt(2*np.pi))*np.exp(-0.5*((x-self.mean)/self.std)**2)

    def logpdf(self, x):
        return -np.log(self.std*np.sqrt(2*np.pi))-0.5*((x-self.mean)/self.std)**2

    def cdf(self, x):
        return 0.5*(1 + erf((x-self.mean)/(self.std*np.sqrt(2))))

    def _sample(self,N=1, rng=None):

        """
        Draw sample(s) from distrubtion
        
        Example
        -------
        p = cuqi.distribution.Normal(mean=2, std=1) #Define distribution
        s = p.sample() #Sample from distribution
        

        Returns
        -------
        Generated sample(s)

        """

        if rng is not None:
            s =  rng.normal(self.mean, self.std, (N,self.dim)).T
        else:
            s = np.random.normal(self.mean, self.std, (N,self.dim)).T
        return s



# ========================================================================
class Gamma(Distribution):

    def __init__(self, shape=None, rate=None, **kwargs):
        # Init from abstract distribution class
        if "geometry" not in kwargs.keys() or kwargs["geometry"] is None:
            #TODO: handle the case when self.shape or self.rate = None because len(None) = 1
            kwargs["geometry"] = max(np.size(shape),np.size(rate))
        super().__init__(**kwargs) 
        self.is_symmetric = False

        # Init specific to this distribution
        self.shape = shape
        self.rate = rate     

    @property
    def dim(self):
        return self.geometry.dim

    @property
    def scale(self):
        return 1/self.rate

    def pdf(self, x):
        # sps.gamma.pdf(x, a=self.shape, loc=0, scale=self.scale)
        # (self.rate**self.shape)/(gamma(self.shape)) * (x**(self.shape-1)*np.exp(-self.rate*x))
        return np.exp(self.logpdf(x))

    def logpdf(self, x):
        # sps.gamma.logpdf(x, a=self.shape, loc=0, scale=self.scale)
        return (self.shape*np.log(self.rate)-loggamma(self.shape)) + ((self.shape-1)*np.log(x) - self.rate*x)

    def cdf(self, x):
        # sps.gamma.cdf(x, a=self.shape, loc=0, scale=self.scale)
        return gammainc(self.shape, self.rate*x)

    def _sample(self, N, rng=None):
        if rng is not None:
            return rng.gamma(shape=self.shape, scale=self.scale, size=(N))
        else:
            return np.random.gamma(shape=self.shape, scale=self.scale, size=(N))

# ========================================================================
class GaussianCov(Distribution): # TODO: super general with precisions
    """
    General Gaussian probability distribution. Generates instance of cuqi.distribution.GaussianCov

    
    Parameters
    ------------
    mean: Mean of distribution. Can be a scalar or 1d numpy array
    cov: Covariance of distribution. Can be a scalar, 1d numpy array (assumes diagonal elements), or 2d numpy array.
    
    Methods
    -----------
    sample: generate one or more random samples
    pdf: evaluate probability density function
    logpdf: evaluate log probability density function
    cdf: evaluate cumulatiuve probability function
    
    Example
    -----------
    # Generate an i.i.d. n-dim Gaussian with zero mean and 2 variance.
    x = cuqi.distribution.Normal(mean=np.zeros(n), cov=2)
    """
    def __init__(self, mean=None, cov=None):
        self.mean = force_ndarray(mean,flatten=True) #Enforce vector shape
        self.cov = force_ndarray(cov)

    @property
    def cov(self):
        return self._cov

    @cov.setter
    def cov(self, value):
        self._cov = value
        if (value is not None) and (not callable(value)):
            prec, sqrtprec, logdet, rank = self.get_prec_from_cov(value)
            self._prec = prec
            self._sqrtprec = sqrtprec
            self._logdet = logdet
            self._rank = rank

    @property
    def dim(self):
        return max(len(self.mean),self.cov.shape[0])

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
                rank = len(d)
                logdet = np.sum(np.log(d))
                prec = sqrtprec @ sqrtprec.T

        return prec, sqrtprec, logdet, rank     

    def logpdf(self, x):
        dev = x - self.mean
        mahadist = np.sum(np.square(dev @ self.sqrtprec), axis=-1)
        return -0.5*(self.rank*np.log(2*np.pi) + self.logdet + mahadist)

    def cdf(self, x1):   # TODO
        return sps.multivariate_normal.cdf(x1, self.mean, self.cov)

    def gradient(self, val, **kwargs):
        if not callable(self.mean): # for prior
            return -self.prec @ (val - self.mean)
        elif hasattr(self.mean,"gradient"): # for likelihood
            model = self.mean
            dev = val - model.forward(**kwargs)
            return self.prec @ model.gradient(dev)
        else:
            warnings.warn('Gradient not implemented for {}'.format(type(self.mean)))

    def _sample(self, N=1, rng=None):
        # If scalar or vector cov use numpy normal
        if (self.cov.shape[0] == 1) or (not issparse(self.cov) and self.cov.shape[0] == np.size(self.cov)): 
            if rng is not None:
                s = rng.normal(self.mean, self.cov, (N,self.dim)).T
            else:
                s = np.random.normal(self.mean, self.cov, (N,self.dim)).T
            return s        
        else:
            if rng is not None:
                s = rng.multivariate_normal(self.mean, self.cov, N).T
            else:
                s = np.random.multivariate_normal(self.mean, self.cov, N).T
            return s


# ========================================================================
class Gaussian(Distribution): #ToDo. Make Gaussian init consistant

    def __init__(self, mean, std, corrmat=None,**kwargs):
        # Init from abstract distribution class
        if corrmat is None:
            corrmat = np.eye(len(mean))
        dim = len(np.diag(corrmat))   #TODO: handle the case when corrmat = None because len(None) = 1
        if "geometry" not in kwargs.keys() or kwargs["geometry"] is None:
            kwargs["geometry"] = dim 
        super().__init__(**kwargs)
        self.is_symmetric = True #TODO: change once we call the super

        self.mean = mean
        self.std = std
        self.R = corrmat

        # self = sps.multivariate_normal(mean, (std**2)*corrmat)

        # pre-computations (covariance and determinants)
        if isinstance(std, (list, tuple, np.ndarray)):
            self.Sigma = np.diag(std) @ (corrmat @ np.diag(std))   # covariance
            isdiag = np.count_nonzero(corrmat - np.diag(np.diagonal(corrmat)))
            if (isdiag == 0):    # uncorrelated
                self.det = np.prod(std**2)
                self.logdet = sum(2*np.log(std))
                self.L = np.linalg.cholesky(self.Sigma)
            else:
                self.det = np.linalg.det(self.Sigma)
                self.L = np.linalg.cholesky(self.Sigma)
                self.logdet = 2*sum(np.log(np.diag(self.L)))  # only for PSD matrices
        else:
            self.Sigma = np.diag(std*np.ones(self.dim)) @ (corrmat @ np.diag(std*np.ones(self.dim)))   # covariance
            isdiag = np.count_nonzero(corrmat - np.diag(np.diagonal(corrmat)))
            if (isdiag == 0):   # uncorrelated
                self.det = std**(2*self.dim)
                self.logdet = 2*self.dim*np.log(std)
                self.L = np.linalg.cholesky(self.Sigma)
            else:
                self.det = std**(2*self.dim) * np.linalg.det(corrmat)
                self.L = np.linalg.cholesky(self.Sigma)
                self.logdet = 2*sum(np.log(np.diag(self.L)))  # only for PSD matrices

        # inverse of Cholesky
        self.Linv = np.linalg.inv(self.L)   

        # Compute decomposition such that Q = U @ U.T
        # self.Sigmainv = np.linalg.inv(self.Sigma)   # precision matrix
        # s, u = eigh(self.Q, lower=True, check_finite=True)
        # s_pinv = np.array([0 if abs(x) <= 1e-5 else 1/x for x in s], dtype=float)
        # self.U = u @ np.diag(np.sqrt(s_pinv))

    @property
    def dim(self):
        return self.geometry.dim

    def logpdf(self, x1, *x2): #TODO use cond dist to handle this kind of input..
        if callable(self.mean):
            mu = self.mean(x2[0])   # mean is variable
        else:
            mu = self.mean       # mean is fix
        xLinv = (x1 - mu) @ self.Linv.T
        quadform = np.sum(np.square(xLinv), 1) if (len(xLinv.shape) > 1) else np.sum(np.square(xLinv))
        # = sps.multivariate_normal.logpdf(x1, mu, self.Sigma)
        return -0.5*(self.logdet + quadform + self.dim*np.log(2*np.pi))

    def pdf(self, x1, *x2):
        # = sps.multivariate_normal.pdf(x1, self.mean, self.Sigma)
        return np.exp(self.logpdf(x1, *x2))

    def cdf(self, x1):   # TODO
        return sps.multivariate_normal.cdf(x1, self.mean, self.Sigma)

    def gradient(self, x):
        if not callable(self.mean):
            return self.Sigmainv@(x-self.mean)

    def _sample(self, N=1, rng=None):

        if rng is not None:
            s = rng.multivariate_normal(self.mean, self.Sigma, N).T
        else:
            s = np.random.multivariate_normal(self.mean, self.Sigma, N).T
            
        return s



# ========================================================================
class GMRF(Gaussian):
        
    def __init__(self, mean, prec, N, dom, BCs, **kwargs): 
        if dom == 1:
            dim = N 
        elif (dom==2):
            dim = N**2
        if "geometry" not in kwargs.keys() or kwargs["geometry"] is None:
            kwargs["geometry"] = dim 
        super(Gaussian, self).__init__(**kwargs) #TODO: This calls Distribution __init__, should be replaced by calling Gaussian.__init__ 

        self.mean = mean.reshape(len(mean), 1)
        self.prec = prec
        self.N = N          # partition size
        self.BCs = BCs      # boundary conditions
        
        # BCs: 1D difference matrix 
        one_vec = np.ones(N)
        dgn = np.vstack([-one_vec, one_vec])
        if (BCs == 'zero'):
            locs = [-1, 0]
            Dmat = spdiags(dgn, locs, N+1, N).tocsc()
        elif (BCs == 'periodic'):
            locs = [-1, 0]
            Dmat = spdiags(dgn, locs, N+1, N).tocsc()
            Dmat[-1, 0] = 1
            Dmat[0, -1] = -1
        elif (BCs == 'neumann'):
            locs = [0, 1]
            Dmat = spdiags(dgn, locs, N, N).tocsc()
            Dmat[-1, -1] = 0
        elif (BCs == 'none'):
            Dmat = eye(N, dtype=int)
        else:
            raise TypeError('Unexpected BC type (choose from zero, periodic, neumann or none)')
        
        # structure matrix
        if (dom == 1):
            self.D = Dmat
            self.L = (Dmat.T @ Dmat).tocsc()
        elif (dom == 2):            
            I = eye(N, dtype=int)
            Ds = kron(I, Dmat)
            Dt = kron(Dmat, I)
            self.D = vstack([Ds, Dt])
            self.L = ((Ds.T @ Ds) + (Dt.T @ Dt)).tocsc()

        self.is_symmetric = True #TODO: change once we call the super   

        # work-around to compute sparse Cholesky
        def sparse_cholesky(A):
            # https://gist.github.com/omitakahiro/c49e5168d04438c5b20c921b928f1f5d
            LU = splinalg.splu(A, diag_pivot_thresh=0, permc_spec='natural') # sparse LU decomposition
  
            # check the matrix A is positive definite
            if (LU.perm_r == np.arange(self.dim)).all() and (LU.U.diagonal() > 0).all(): 
                return LU.L @ (diags(LU.U.diagonal()**0.5))
            else:
                raise TypeError('The matrix is not positive semi-definite')
        
        # compute Cholesky and det
        if (BCs == 'zero'):    # only for PSD matrices
            self.rank = self.dim
            self.chol = sparse_cholesky(self.L)
            self.logdet = 2*sum(np.log(self.chol.diagonal()))
            # L_cholmod = cholesky(self.L, ordering_method='natural')
            # self.chol = L_cholmod
            # self.logdet = L_cholmod.logdet()
            # 
            # np.log(np.linalg.det(self.L.todense()))
        elif (BCs == 'periodic') or (BCs == 'neumann'):
            self.rank = self.dim - 1   #np.linalg.matrix_rank(self.L.todense())
            self.chol = sparse_cholesky(self.L + np.sqrt(eps)*eye(self.dim, dtype=int))
            if (self.dim > 5000):  # approximate to avoid 'excesive' time
                self.logdet = 2*sum(np.log(self.chol.diagonal()))
            else:
                # eigval = eigvalsh(self.L.todense())
                self.L_eigval = splinalg.eigsh(self.L, self.rank, which='LM', return_eigenvectors=False)
                self.logdet = sum(np.log(self.L_eigval))


    @property 
    def dim(self):
        return self.geometry.dim

    def logpdf(self, x):
        const = 0.5*(self.rank*(np.log(self.prec)-np.log(2*np.pi)) + self.logdet)
        y = const - 0.5*( self.prec*((x-self.mean).T @ (self.L @ (x-self.mean))) )
        y = np.diag(y)
        # = sps.multivariate_normal.logpdf(x.T, self.mean.flatten(), np.linalg.inv(self.prec*self.L.todense()))
        return y

    def pdf(self, x):
        # = sps.multivariate_normal.pdf(x.T, self.mean.flatten(), np.linalg.inv(self.prec*self.L.todense()))
        return np.exp(self.logpdf(x))

    def gradient(self, x):
        if not callable(self.mean):
            return (self.prec*self.L) @ (x-self.mean)

    def sample(self, Ns=1, rng=None):
        if (self.BCs == 'zero'):

            if rng is not None:
                xi = rng.standard_normal((self.dim, Ns))   # standard Gaussian
            else:
                xi = np.random.randn(self.dim, Ns)   # standard Gaussian

            if Ns == 1:
                s = self.mean.flatten() + (1/np.sqrt(self.prec))*splinalg.spsolve(self.chol.T, xi)
            else:
                s = self.mean + (1/np.sqrt(self.prec))*splinalg.spsolve(self.chol.T, xi)
            # s = self.mean + (1/np.sqrt(self.prec))*L_cholmod.solve_Lt(xi, use_LDLt_decomposition=False) 
                        
        elif (self.BCs == 'periodic'):

            if rng is not None:
                xi = rng.standard_normal((self.dim, Ns)) + 1j*rng.standard_normal((self.dim, Ns))
            else:
                xi = np.random.randn(self.dim, Ns) + 1j*np.random.randn(self.dim, Ns)
            
            F = dft(self.dim, scale='sqrtn')   # unitary DFT matrix
            # eigv = eigvalsh(self.L.todense()) # splinalg.eigsh(self.L, self.rank, return_eigenvectors=False)           
            eigv = np.hstack([self.L_eigval, self.L_eigval[-1]])  # repeat last eigval to complete dim
            L_sqrt = diags(np.sqrt(eigv)) 
            s = self.mean + (1/np.sqrt(self.prec))*np.real(F.conj() @ splinalg.spsolve(L_sqrt, xi))
            # L_sqrt = pinvh(np.diag(np.sqrt(eigv)))
            # s = self.mean + (1/np.sqrt(self.prec))*np.real(F.conj() @ (L_sqrt @ xi))
            
        elif (self.BCs == 'neumann'):

            if rng is not None:
                xi = rng.standard_normal((self.D.shape[0], Ns))   # standard Gaussian
            else:
                xi = np.random.randn(self.D.shape[0], Ns)   # standard Gaussian
            
            s = self.mean + (1/np.sqrt(self.prec))* \
                splinalg.spsolve(self.chol.T, (splinalg.spsolve(self.chol, (self.D.T @ xi)))) 
        else:
            raise TypeError('Unexpected BC type (choose from zero, periodic, neumann or none)')

        return s
        


# ========================================================================
class Laplace_diff(object):

    def __init__(self, location, scale, bndcond):
        self.loc = location
        self.scale = scale
        self.dim = len(location)
        self.bnd = bndcond

        # finite difference matrix
        one_vec = np.ones(self.dim)
        diags = np.vstack([-one_vec, one_vec])
        if (bndcond == 'zero'):
            locs = [-1, 0]
            Dmat = spdiags(diags, locs, self.dim+1, self.dim)
        elif (bndcond == 'periodic'):
            locs = [-1, 0]
            Dmat = spdiags(diags, locs, self.dim+1, self.dim).tocsr()
            Dmat[-1, 0] = 1
            Dmat[0, -1] = -1
        elif (bndcond == 'neumann'):
            locs = [0, 1]
            Dmat = spdiags(diags, locs, self.dim-1, self.dim)
        elif (bndcond == 'backward'):
            locs = [0, -1]
            Dmat = spdiags(diags, locs, self.dim, self.dim).tocsr()
            Dmat[0, 0] = 1
        elif (bndcond == 'none'):
            Dmat = eye(self.dim)
        self.D = Dmat
        self.is_symmetric = None #TODO: update

    def pdf(self, x):
        Dx = self.D @ (x-self.loc)  # np.diff(X)
        return (1/(2*self.scale))**(len(Dx)) * np.exp(-np.linalg.norm(Dx, ord=1, axis=0)/self.scale)

    def logpdf(self, x):
        Dx = self.D @ (x-self.loc)
        return len(Dx)*(-(np.log(2)+np.log(self.scale))) - np.linalg.norm(Dx, ord=1, axis=0)/self.scale

    # def cdf(self, x):   # TODO
    #     return 1/2 + 1/2*np.sign(x-self.loc)*(1-np.exp(-np.linalg.norm(x, ord=1, axis=0)/self.scale))

    # def sample(self):   # TODO
    #     p = np.random.rand(self.dim)
    #     return self.loc - self.scale*np.sign(p-1/2)*np.log(1-2*abs(p-1/2))


class Uniform(Distribution):


    def __init__(self, low=None, high=None, **kwargs):
        """
        Parameters
        ----------
        low : float or array_like of floats
            Lower bound(s) of the uniform distribution.
        high : float or array_like of floats 
            Upper bound(s) of the uniform distribution.
        """
        # Init from abstract distribution class
        if "geometry" not in kwargs.keys() or kwargs["geometry"] is None:
            kwargs["geometry"] = max(np.size(low),np.size(high)) 
        super().__init__(**kwargs) #TODO: This calls Distribution __init__, should be replaced by calling Gaussian.__init__      

        # Init specific to this distribution
        self.low = low
        self.high = high  
        self.is_symmetric = True       


    @property 
    def dim(self):
        return self.geometry.dim


    def logpdf(self, x):
        diff = self.high -self.low
        if isinstance(diff, (list, tuple, np.ndarray)): 
            v= np.prod(diff)
        else:
            v = diff
        return np.log(1.0/v) 

    def _sample(self,N=1, rng=None):

        if rng is not None:
            s = rng.uniform(self.low, self.high, (N,self.dim)).T
        else:
            s = np.random.uniform(self.low, self.high, (N,self.dim)).T

        return s

# ========================================================================
class Posterior(Distribution):
        
    def __init__(self, likelihood, prior, data, **kwargs):
        # Init from abstract distribution class
        self.likelihood = likelihood
        self.prior = prior 
        self.data = data
        self.dim = prior.dim
        super().__init__(**kwargs)

    def logpdf(self,x):

        return self.likelihood(x=x).logpdf(self.data)+ self.prior.logpdf(x)

    def _sample(self,N=1,rng=None):
        raise Exception("'Posterior.sample' is not defined. Sampling can be performed with the 'sampler' module.")

class UserDefinedDistribution(Distribution):

    def __init__(self, logpdf_func, **kwargs):

        # Init from abstract distribution class
        super().__init__(**kwargs)

        if not callable(logpdf_func): raise ValueError("logpdf_func should be callable")
        self.logpdf_func = logpdf_func

    def logpdf(self, x):
        return self.logpdf_func(x)

    def _sample(self,N=1,rng=None):
        raise Exception("'Generic.sample' is not defined. Sampling can be performed with the 'sampler' module.")


class DistributionGallery(UserDefinedDistribution):

    def __init__(self, distribution_name,**kwargs):
        # Init from abstract distribution class
        if distribution_name is "CalSom91":
            #TODO: user can specify sig and delta
            self.dim = 2
            self.sig = 0.1
            self.delta = 1
            logpdf_func = self._CalSom91_logpdf_func
        elif distribution_name is "BivariateGaussian":
            #TODO: user can specify Gaussain input
            #TODO: Keep Gaussian distribution other functionalities (e.g. _sample)
            self.dim = 2
            mu = np.zeros(self.dim)
            sigma = np.linspace(0.5, 1, self.dim)
            R = np.array([[1.0, .9 ],[.9, 1]])
            dist = Gaussian(mu, sigma, R)
            self._sample = dist._sample
            logpdf_func = dist.logpdf

        super().__init__(logpdf_func, **kwargs)

    def _CalSom91_logpdf_func(self,x):
        if len(x.shape) == 1:
            x = x.reshape( (1,2))
        return -1/(2*self.sig**2)*(np.sqrt(x[:,0]**2+ x[:,1]**2) -1 )**2 -1/(2*self.delta**2)*(x[:,1]-1)**2


