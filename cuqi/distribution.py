import numpy as np
import scipy.stats as sps
from scipy.special import erf, loggamma, gammainc
from scipy.sparse import diags, eye, identity, issparse
from scipy.sparse import linalg as splinalg
from scipy.linalg import eigh, dft, cho_solve, cho_factor, eigvals, lstsq
from cuqi.samples import Samples, CUQIarray
from cuqi.geometry import _DefaultGeometry, Geometry
from cuqi.utilities import force_ndarray, getNonDefaultArgs, get_indirect_attributes
import warnings
from cuqi.operator import FirstOrderFiniteDifference, PrecisionFiniteDifference
from abc import ABC, abstractmethod
from copy import copy
from functools import partial
import warnings


# ========== Abstract distribution class ===========
class Distribution(ABC):

    def __init__(self,name=None, geometry=None, is_symmetric=None):
        if not isinstance(name,str) and name is not None:
            raise ValueError("Name must be a string or None")
        self.name = name
        self.is_symmetric = is_symmetric
        self.geometry = geometry

    @property
    @abstractmethod
    def dim(self):
        pass

    @property
    def geometry(self):
        if self.dim != self._geometry.dim:
            if isinstance(self._geometry,_DefaultGeometry):
                self.geometry = self.dim
            else:
                raise Exception("Distribution Geometry attribute is not consistent with the distribution dimension ('dim')")
        return self._geometry

    @geometry.setter
    def geometry(self,value):
        if isinstance(value, (int,np.integer)) or value is None:
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
        #for key, value in vars(self).items():
        #    if isinstance(value,Distribution) or callable(value):
        #        raise NotImplementedError("Parameter {} is {}. Parameter must be a fixed value.".format(key,value))

        # Get samples from the distribution sample method
        s = self._sample(N,*args,**kwargs)

        #Store samples in cuqi samples object if more than 1 sample
        if N==1:
            if len(s) == 1 and isinstance(s,np.ndarray): #Extract single value from numpy array
                s = s.ravel()[0]
            else:
                s = s.flatten()
            s = CUQIarray(s, geometry=self.geometry)
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
                elif callable(attr_val) and kw_key in getNonDefaultArgs(attr_val):
                    val_found = 1
            if val_found == 0:
                raise ValueError("The keyword {} is not part of any attribute or non-default argument to any function of this distribution.".format(kw_key))


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

                accepted_keywords = getNonDefaultArgs(attr_val)
                remaining_keywords = copy(accepted_keywords)

                # Builds dict with arguments to call attribute with
                attr_args = {}
                for kw_key, kw_val in kwargs.items():
                    if kw_key in accepted_keywords:
                        attr_args[kw_key] = kw_val
                        remaining_keywords.remove(kw_key)

                # If any keywords matched call with those and store output in the new dist
                if len(attr_args)==len(accepted_keywords):  #All keywords found
                    setattr(new_dist,attr_key,attr_val(**attr_args))
                elif len(attr_args)>0:                      #Some keywords found
                    # Define new function where the conditioned keywords are defined
                    func = partial(attr_val,**attr_args)
                    setattr(new_dist,attr_key,func)

        return new_dist


    def get_conditioning_variables(self):
        attributes = []
        ignore_attributes = ['name', 'is_symmetric']
        for key,value in vars(self).items():
            if vars(self)[key] is None and key[0] != '_' and key not in ignore_attributes:
                attributes.append(key)
        return attributes + get_indirect_attributes(self) 

# ========================================================================
class Cauchy_diff(Distribution):

    def __init__(self, location, scale, bc_type,**kwargs):
        # Init from abstract distribution class
        super().__init__(**kwargs) 
        
        self.location = location
        self.scale = scale
        self._bc_type = bc_type

        self._diff_op = FirstOrderFiniteDifference(self.dim, bc_type=bc_type)

    @property
    def dim(self): 
        #TODO: handle the case when self.loc = None because len(None) = 1
        return len(self.location)

    def logpdf(self, x):
        Dx = self._diff_op @ (x-self.location)
        # g_logpr = (-2*Dx/(Dx**2 + gamma**2)) @ D
        return -len(Dx)*np.log(np.pi) + sum(np.log(self.scale) - np.log(Dx**2 + self.scale**2))
    
    def gradient(self, val, **kwargs):
        if not callable(self.location): # for prior
            diff = self._diff_op._matrix @ val
            return (-2*diff/(diff**2+self.scale**2)) @ self._diff_op._matrix
        else:
            warnings.warn('Gradient not implemented for {}'.format(type(self.location)))

    def _sample(self,N=1,rng=None):
        raise NotImplementedError("'Cauchy_diff.sample' is not implemented. Sampling can be performed with the 'sampler' module.")

    # def cdf(self, x):   # TODO
    #     return 1/np.pi * np.atan((x-self.loc)/self.scale)

    # def sample(self):   # TODO
    #     return self.loc + self.scale*np.tan(np.pi*(np.random.rand(self.dim)-1/2))


# ========================================================================
class Normal(Distribution):
    """
    Normal probability distribution. Generates instance of cuqi.distribution.Normal. The variables of this distribution are iid.

    
    Parameters
    ------------
    mean: mean of distribution
    std: standard deviation
    
    Methods
    -----------
    sample: generate one or more random samples
    pdf: evaluate probability density function
    logpdf: evaluate log probability density function
    cdf: evaluate cumulative probability function
    
    Example
    -----------
    #Generate Normal with mean 2 and standard deviation 1
    p = cuqi.distribution.Normal(mean=2, std=1)
    """
    def __init__(self, mean=None, std=None, is_symmetric=True, **kwargs):
        # Init from abstract distribution class
        super().__init__(is_symmetric=is_symmetric, **kwargs)  

        # Init specific to this distribution
        self.mean = mean
        self.std = std


    @property
    def dim(self): 
        #TODO: handle the case when self.mean or self.std = None because len(None) = 1
        if self.mean is None and self.std is None:
            return None
        else:
            return max(np.size(self.mean),np.size(self.std))

    def pdf(self, x):
        return 1/(self.std*np.sqrt(2*np.pi))*np.exp(-0.5*((x-self.mean)/self.std)**2)

    def logpdf(self, x):
        return -np.log(self.std*np.sqrt(2*np.pi))-0.5*((x-self.mean)/self.std)**2

    def cdf(self, x):
        return 0.5*(1 + erf((x-self.mean)/(self.std*np.sqrt(2))))

    def _sample(self,N=1, rng=None):

        """
        Draw sample(s) from distribution
        
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

    def __init__(self, shape=None, rate=None, is_symmetric=False, **kwargs):
        # Init from abstract distribution class
        super().__init__(is_symmetric=is_symmetric,**kwargs) 

        # Init specific to this distribution
        self.shape = shape
        self.rate = rate     

    @property
    def dim(self):
        #TODO: handle the case when self.shape or self.rate = None because len(None) = 1
        return max(np.size(self.shape),np.size(self.rate))

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
    cdf: evaluate cumulative probability function
    
    Example
    -----------
    # Generate an i.i.d. n-dim Gaussian with zero mean and 2 variance.
    x = cuqi.distribution.Normal(mean=np.zeros(n), cov=2)
    """
    def __init__(self, mean=None, cov=None, is_symmetric=True, **kwargs):
        super().__init__(is_symmetric=is_symmetric, **kwargs) 

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
        if not hasattr(self.mean,"__len__"): #TODO: this need to be generalized for all dim properties.
            return self.cov.shape[0] 
        else:
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

    @property
    def dim(self):
        #TODO: handle the case when self.mean or self.sqrtprec = None because len(None) = 1
        return max(np.size(self.mean),np.shape(self.sqrtprec)[1])

    def _sample(self, N):
        if issparse(self.sqrtprec):        
            # sample using x = mean + pseudoinverse(sqrtprec)*eps, where eps is N(0,1)
            samples = self.mean[:, None] + splinalg.spsolve(self.sqrtprec, np.random.randn(np.shape(self.sqrtprec)[0], N))
        else:
            # sample using x = mean + pseudoinverse(sqrtprec)*eps, where eps is N(0,1)
            samples = self.mean[:, None] + lstsq(self.sqrtprec, np.random.randn(np.shape(self.sqrtprec)[0], N), cond = 1e-14)[0]
        return samples

    def logpdf(self, x):
        # Sqrtprec diagonal
        if (issparse(self.sqrtprec) and self.sqrtprec.format == 'dia'): 
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
                s = eigvals(self.sqrtprec.T@self.sqrtprec)
                d = s[s > eps]
                rank = len(d)
                logdet = np.sum(np.log(d))

        dev = x - self.mean
        mahadist = np.sum(np.square(self.sqrtprec @ dev), axis=0)
        # rank(prec) = rank(sqrtprec.T*sqrtprec) = rank(sqrtprec)
        # logdet can also be pseudo-determinant, defined as the product of non-zero eigenvalues
        return -0.5*(rank*np.log(2*np.pi) - logdet + mahadist)

class GaussianPrec(Distribution):

    def __init__(self,mean,prec,is_symmetric=True,**kwargs):
        super().__init__(is_symmetric=is_symmetric, **kwargs) 

        self.mean = force_ndarray(mean,flatten=True) #Enforce vector shape
        self.prec = force_ndarray(prec)

    def _sample(self,N=1):
        # Pre-allocate
        s = np.zeros((self.dim,N))
        
        # Cholesky factor of precision
        R,low = cho_factor(self.prec)
        
        # Sample
        for i in range(N):
            s[:,i] = self.mean+cho_solve((R,low),np.random.randn(self.dim))
        return s

    def logpdf(self,x):
        raise NotImplementedError("Logpdf of GaussianPrec is not yet implemented.")

    @property
    def dim(self):
        return max(len(self.mean),self.prec.shape[0])

class Gaussian(GaussianCov):

    def __init__(self, mean=None, std=None, corrmat=None, is_symmetric=True, **kwargs):
        print("Initializing as GaussianCov. Mutable attributes: mean, cov")
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


# ========================================================================
class GMRF(Distribution):
    """
        Parameters
        ----------
        partition_size : int
            The dimension of the distribution in one physical dimension. 

        physical_dim : int
            The physical dimension of what the distribution represents (can take the values 1 or 2).
    """
        
    def __init__(self, mean, prec, partition_size, physical_dim, bc_type, is_symmetric=True, **kwargs): 
        super().__init__(is_symmetric=is_symmetric, **kwargs) #TODO: This calls Distribution __init__, should be replaced by calling Gaussian.__init__ 

        self.mean = mean.reshape(len(mean), 1)
        self.prec = prec
        self._partition_size = partition_size          # partition size
        self._bc_type = bc_type      # boundary conditions
        self._physical_dim = physical_dim
        if physical_dim == 1: 
            num_nodes = (partition_size,) 
        else:
            num_nodes = (partition_size,partition_size)

        self._prec_op = PrecisionFiniteDifference( num_nodes, bc_type= bc_type, order =1) 
        self._diff_op = self._prec_op._diff_op      
            
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
        if (bc_type == 'zero'):    # only for PSD matrices
            self._rank = self.dim
            self._chol = sparse_cholesky(self._prec_op.get_matrix())
            self._logdet = 2*sum(np.log(self._chol.diagonal()))
            # L_cholmod = cholesky(self.L, ordering_method='natural')
            # self.chol = L_cholmod
            # self.logdet = L_cholmod.logdet()
            # 
            # np.log(np.linalg.det(self.L.todense()))
        elif (bc_type == 'periodic') or (bc_type == 'neumann'):
            eps = np.finfo(float).eps
            self._rank = self.dim - 1   #np.linalg.matrix_rank(self.L.todense())
            self._chol = sparse_cholesky(self._prec_op + np.sqrt(eps)*eye(self.dim, dtype=int))
            if (self.dim > 5000):  # approximate to avoid 'excesive' time
                self._logdet = 2*sum(np.log(self._chol.diagonal()))
            else:
                # eigval = eigvalsh(self.L.todense())
                self._L_eigval = splinalg.eigsh(self._prec_op.get_matrix(), self._rank, which='LM', return_eigenvectors=False)
                self._logdet = sum(np.log(self._L_eigval))


    @property 
    def dim(self):  
        if self._physical_dim == 1:
            return self._partition_size 
        elif self._physical_dim==2:
            return self._partition_size**2
        raise ValueError("attribute dom can be either 1 or 2")

    def logpdf(self, x):
        const = 0.5*(self._rank*(np.log(self.prec)-np.log(2*np.pi)) + self._logdet)
        y = const - 0.5*( self.prec*((x-self.mean).T @ (self._prec_op @ (x-self.mean))) )
        y = np.diag(y)
        # = sps.multivariate_normal.logpdf(x.T, self.mean.flatten(), np.linalg.inv(self.prec*self.L.todense()))
        return y

    def pdf(self, x):
        # = sps.multivariate_normal.pdf(x.T, self.mean.flatten(), np.linalg.inv(self.prec*self.L.todense()))
        return np.exp(self.logpdf(x))

    def gradient(self, x):
        if not callable(self.mean):
            return (self.prec*self._prec_op) @ (x-self.mean)

    def _sample(self, N=1, rng=None):
        if (self._bc_type == 'zero'):

            if rng is not None:
                xi = rng.standard_normal((self.dim, N))   # standard Gaussian
            else:
                xi = np.random.randn(self.dim, N)   # standard Gaussian

            if N == 1:
                s = self.mean.flatten() + (1/np.sqrt(self.prec))*splinalg.spsolve(self._chol.T, xi)
            else:
                s = self.mean + (1/np.sqrt(self.prec))*splinalg.spsolve(self._chol.T, xi)
            # s = self.mean + (1/np.sqrt(self.prec))*L_cholmod.solve_Lt(xi, use_LDLt_decomposition=False) 
                        
        elif (self._bc_type == 'periodic'):

            if rng is not None:
                xi = rng.standard_normal((self.dim, N)) + 1j*rng.standard_normal((self.dim, N))
            else:
                xi = np.random.randn(self.dim, N) + 1j*np.random.randn(self.dim, N)
            
            F = dft(self.dim, scale='sqrtn')   # unitary DFT matrix
            # eigv = eigvalsh(self.L.todense()) # splinalg.eigsh(self.L, self.rank, return_eigenvectors=False)           
            eigv = np.hstack([self._L_eigval, self._L_eigval[-1]])  # repeat last eigval to complete dim
            L_sqrt = diags(np.sqrt(eigv)) 
            s = self.mean + (1/np.sqrt(self.prec))*np.real(F.conj() @ splinalg.spsolve(L_sqrt, xi))
            # L_sqrt = pinvh(np.diag(np.sqrt(eigv)))
            # s = self.mean + (1/np.sqrt(self.prec))*np.real(F.conj() @ (L_sqrt @ xi))
            
        elif (self._bc_type == 'neumann'):

            if rng is not None:
                xi = rng.standard_normal((self._diff_op.shape[0], N))   # standard Gaussian
            else:
                xi = np.random.randn(self._diff_op.shape[0], N)   # standard Gaussian
            
            s = self.mean + (1/np.sqrt(self.prec))* \
                splinalg.spsolve(self._chol.T, (splinalg.spsolve(self._chol, (self._diff_op.T @ xi)))) 
        else:
            raise TypeError('Unexpected BC type (choose from zero, periodic, neumann or none)')

        return s
        


# ========================================================================
class Laplace_diff(Distribution):

    def __init__(self, location, scale, bc_type, **kwargs):
        # Init from abstract distribution class
        super().__init__(**kwargs) 

        self.location = location
        self.scale = scale
        self._bc_type = bc_type

        # finite difference matrix
        self._diff_op = FirstOrderFiniteDifference(self.dim, bc_type=bc_type)


    @property
    def dim(self):
        #TODO: handle the case when self.loc is None 
        return len(self.location)

    def pdf(self, x):
        Dx = self._diff_op @ (x-self.location)  # np.diff(X)
        return (1/(2*self.scale))**(len(Dx)) * np.exp(-np.linalg.norm(Dx, ord=1, axis=0)/self.scale)

    def logpdf(self, x):
        Dx = self._diff_op @ (x-self.location)
        return len(Dx)*(-(np.log(2)+np.log(self.scale))) - np.linalg.norm(Dx, ord=1, axis=0)/self.scale

    def _sample(self,N=1,rng=None):
        raise NotImplementedError("'Laplace_diff.sample' is not implemented. Sampling can be performed with the 'sampler' module.")

    # def cdf(self, x):   # TODO
    #     return 1/2 + 1/2*np.sign(x-self.loc)*(1-np.exp(-np.linalg.norm(x, ord=1, axis=0)/self.scale))

    # def sample(self):   # TODO
    #     p = np.random.rand(self.dim)
    #     return self.loc - self.scale*np.sign(p-1/2)*np.log(1-2*abs(p-1/2))


class Uniform(Distribution):


    def __init__(self, low=None, high=None, is_symmetric=True, **kwargs):
        """
        Parameters
        ----------
        low : float or array_like of floats
            Lower bound(s) of the uniform distribution.
        high : float or array_like of floats 
            Upper bound(s) of the uniform distribution.
        """
        # Init from abstract distribution class
        super().__init__(is_symmetric=is_symmetric, **kwargs)       

        # Init specific to this distribution
        self.low = low
        self.high = high      


    @property 
    def dim(self):
        #TODO: hanlde the case when high and low are None
        return max(np.size(self.low),np.size(self.high)) 


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
        if 'geometry' not in kwargs.keys(): 
            kwargs["geometry"]=prior.geometry
        super().__init__(**kwargs)

    @property
    def dim(self):
        return self.prior.dim

    @property
    def geometry(self):
        return self.prior.geometry

    @geometry.setter
    def geometry(self, value):
        if value != self.prior.geometry:
            raise ValueError("Posterior and prior geometries are inconsistent.")
        # no need to actually set geometry because self.geometry returns self.prior.geometry


    def logpdf(self,x):

        return self.likelihood(x=x).logpdf(self.data)+ self.prior.logpdf(x)

    def gradient(self, x):
        return self.likelihood.gradient(self.data, x=x)+ self.prior.gradient(x)        

    def _sample(self,N=1,rng=None):
        raise Exception("'Posterior.sample' is not defined. Sampling can be performed with the 'sampler' module.")

class UserDefinedDistribution(Distribution):

    def __init__(self, logpdf_func, dim, **kwargs):

        # Init from abstract distribution class
        super().__init__(**kwargs)

        if not callable(logpdf_func): raise ValueError("logpdf_func should be callable")
        self.logpdf_func = logpdf_func
        self.dim = dim

    @property
    def dim(self):
        return self._dim

    @dim.setter
    def dim(self, value):
        self._dim = value

    def logpdf(self, x):
        return self.logpdf_func(x)

    def _sample(self,N=1,rng=None):
        raise Exception("'Generic.sample' is not defined. Sampling can be performed with the 'sampler' module.")


class DistributionGallery(UserDefinedDistribution):

    def __init__(self, distribution_name,**kwargs):
        # Init from abstract distribution class
        if distribution_name == "CalSom91":
            #TODO: user can specify sig and delta
            dim = 2
            self.sig = 0.1
            self.delta = 1
            logpdf_func = self._CalSom91_logpdf_func
        elif distribution_name == "BivariateGaussian":
            #TODO: user can specify Gaussain input
            #TODO: Keep Gaussian distribution other functionalities (e.g. _sample)
            dim = 2
            mu = np.zeros(dim)
            sigma = np.linspace(0.5, 1, dim)
            R = np.array([[1.0, .9 ],[.9, 1]])
            dist = Gaussian(mu, sigma, R)
            self._sample = dist._sample
            logpdf_func = dist.logpdf

        super().__init__(logpdf_func, dim, **kwargs)


    def _CalSom91_logpdf_func(self,x):
        if len(x.shape) == 1:
            x = x.reshape( (1,2))
        return -1/(2*self.sig**2)*(np.sqrt(x[:,0]**2+ x[:,1]**2) -1 )**2 -1/(2*self.delta**2)*(x[:,1]-1)**2



# ========================================================================
class Laplace(Distribution):
    """
    The variables of this Laplace distribution are iid.
    """

    def __init__(self, location, prec, **kwargs):

        # Init from abstract distribution class
        super().__init__(**kwargs)

        self.location = location
        self.prec = prec
  
    @property
    def dim(self):
        return np.size(self.location)

    def logpdf(self, x):
        if isinstance(x, (float,int)):
            x = np.array([x])
        return self.dim*(np.log(self.prec/2)) - self.prec*np.linalg.norm(x-self.location,1)

    def _sample(self,N=1,rng=None):
        if rng is not None:
            s =  rng.laplace(self.location, 1.0/self.prec, (N,self.dim))
        else:
            s = np.random.laplace(self.location, 1.0/self.prec, (N,self.dim))
        return s

# ========================================================================
class LMRF(Distribution):
    """
        Parameters
        ----------
        partition_size : int
            The dimension of the distribution in one physical dimension. 

        physical_dim : int
            The physical dimension of what the distribution represents (can take the values 1 or 2).
    """
        
    def __init__(self, mean, prec, partition_size, physical_dim, bc_type, **kwargs):
        super().__init__(**kwargs)
        self.mean = mean.reshape(len(mean), 1)
        self.prec = prec
        self._partition_size = partition_size          # partition size
        self._bc_type = bc_type      # boundary conditions
        self._physical_dim = physical_dim
        if physical_dim == 1: 
            num_nodes = (partition_size,) 
        else:
            num_nodes = (partition_size,partition_size)

        self._diff_op = FirstOrderFiniteDifference( num_nodes, bc_type= bc_type) 

    @property
    def dim(self):
        return self._diff_op.dim

    def logpdf(self, x):

        if self._physical_dim == 1 or self._physical_dim == 2:
            const = self.dim *(np.log(self.prec)-np.log(2)) 
            y = const -  self.prec*(np.linalg.norm(self._diff_op@x, ord=1))
        else:
            raise NotImplementedError
        return y

    def _sample(self, N):
        raise NotImplementedError