import numpy as np
import scipy.stats as sps
from scipy.special import erf, loggamma, gammainc
from scipy.sparse import diags, eye, identity, issparse, vstack
from scipy.sparse import linalg as splinalg
from scipy.linalg import eigh, dft, cho_solve, cho_factor, eigvals, lstsq, cholesky
from cuqi.samples import Samples, CUQIarray
from cuqi.geometry import _DefaultGeometry, Geometry, Image2D, _get_identity_geometries
from cuqi.utilities import force_ndarray, get_writeable_attributes, get_writeable_properties, get_non_default_args, get_indirect_variables, sparse_cholesky
from cuqi.model import Model
from cuqi.likelihood import Likelihood
from cuqi import config
import warnings
from cuqi.operator import FirstOrderFiniteDifference, PrecisionFiniteDifference
from abc import ABC, abstractmethod
from copy import copy
from functools import partial
import warnings


# ========== Abstract distribution class ===========
class Distribution(ABC):
    """ Abstract Base Class for Distributions.

    Handles functionality for pdf evaluation, sampling, geometries and conditioning.
    
    Parameters
    ----------
    name : str, default None
        Name of distribution.
    
    geometry : Geometry, default _DefaultGeometry (or None)
        Geometry of distribution.

    is_symmetric : bool, default None
        Indicator if distribution is symmetric.

    Attributes
    ----------
    dim : int or None
        Dimension of distribution.

    name : str or None
        Name of distribution.
    
    geometry : Geometry or None
        Geometry of distribution.

    is_cond : bool
        Indicator if distribution is conditional.

    Methods
    -------
    pdf():
        Evaluate the probability density function.

    logpdf():
        Evaluate the log probability density function.

    sample():
        Generate one or more random samples.

    get_conditioning_variables():
        Return the conditioning variables of distribution.

    get_mutable_variables():
        Return the mutable variables (attributes and properties) of distribution.

    Notes
    -----
    A distribution can be conditional if one or more mutable variables are unspecified.
    A mutable variable can be unspecified in one of two ways:

    1. The variable is set to None.
    2. The variable is set to a callable function with non-default arguments.

    The conditioning variables of a conditional distribution are then defined to be the
    mutable variable itself (in case 1) or the parameters to the callable function (in case 2).

    """
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

    def __call__(self, *args, **kwargs):
        """ Generate new distribution conditioned on the input arguments. """

        # Store conditioning variables and mutable variables
        cond_vars = self.get_conditioning_variables()
        mutable_vars = self.get_mutable_variables()

        # PARSE ARGS AND ADD TO KWARGS
        if len(args)>0:
            # If no cond_vars we throw error since we cant get order.
            if len(cond_vars)==0:
                raise ValueError("Unable to parse args since this distribution has no conditioning variables. Use keywords to modify mutable variables.")
            ordered_keys = cond_vars # Args follow order of cond. vars
            for index, arg in enumerate(args):
                if ordered_keys[index] in kwargs:
                    raise ValueError(f"{ordered_keys[index]} passed as both argument and keyword argument.\nArguments follow the listed conditioning variable order: {self.get_conditioning_variables()}")
                kwargs[ordered_keys[index]] = arg

        # KEYWORD ERROR CHECK
        for kw_key in kwargs.keys():
            if kw_key not in (mutable_vars+cond_vars):
                raise ValueError("The keyword \"{}\" is not a mutable or conditioning variable of this distribution.".format(kw_key))

        # EVALUATE CONDITIONAL DISTRIBUTION
        new_dist = copy(self) #New cuqi distribution conditioned on the kwargs
        new_dist.name = None  #Reset name to None

        # Go through every mutable variable and assign value from kwargs if present
        for var_key in mutable_vars:

            #If keyword directly specifies new value of variable we simply reassign
            if var_key in kwargs:
                setattr(new_dist, var_key, kwargs.get(var_key))

            # If variable is callable we check if any keyword arguments
            # can be used as arguments to the callable method.
            var_val = getattr(self, var_key) # Get current value of variable
            if callable(var_val):

                accepted_keywords = get_non_default_args(var_val)
                remaining_keywords = copy(accepted_keywords)

                # Builds dict with arguments to call variable with
                var_args = {}
                for kw_key, kw_val in kwargs.items():
                    if kw_key in accepted_keywords:
                        var_args[kw_key] = kw_val
                        remaining_keywords.remove(kw_key)

                # If any keywords matched we evaluate callable variable
                if len(var_args)==len(accepted_keywords):  #All keywords found
                    # Define variable as the output of callable function
                    setattr(new_dist, var_key, var_val(**var_args))

                elif len(var_args)>0:                      #Some keywords found
                    # Define new partial function with partially defined args
                    func = partial(var_val, **var_args)
                    setattr(new_dist, var_key, func)

        return new_dist


    def get_conditioning_variables(self):
        """Return the conditioning variables of this distribution (if any)."""
        
        # Get all mutable variables
        mutable_vars = self.get_mutable_variables()

        # Loop over mutable variables and if None they are conditioning variables
        cond_vars = [key for key in mutable_vars if getattr(self, key) is None]

        # Add any variables defined through callable functions
        cond_vars += get_indirect_variables(self)
        
        return cond_vars

    def get_mutable_variables(self):
        """Return any public variable that is mutable (attribute or property) except those in the ignore_vars list"""
        # Define list of ignored attributes and properties
        ignore_vars = ['name', 'is_symmetric', 'geometry', 'dim']
        
        # Get public attributes
        attributes = get_writeable_attributes(self)

        # Get "public" properties (getter+setter)
        properties = get_writeable_properties(self)

        return [var for var in (attributes+properties) if var not in ignore_vars]

    @property
    def is_cond(self):
        """ Returns True if instance (self) is a conditional distribution. """
        if len(self.get_conditioning_variables()) == 0:
            return False
        else:
            return True

    def to_likelihood(self, data):
        """Convert conditional distribution to a likelihood function given observed data"""
        return Likelihood(self, data)


    def __repr__(self) -> str:
        if self.is_cond is True:
            return "CUQI {}. Conditioning variables {}.".format(self.__class__.__name__,self.get_conditioning_variables())
        else:
            return "CUQI {}.".format(self.__class__.__name__)
# ========================================================================
class Cauchy_diff(Distribution):
    """Cauchy distribution on the difference between neighboring nodes.

    Parameters
    ----------
    location : scalar or ndarray
        The location parameter of the distribution.

    scale : scalar
        The scale parameter of the distribution.

    bc_type : string
        The boundary conditions of the difference operator.

    physical_dim : int
        The physical dimension of what the distribution represents (can take the values 1 or 2).

    Example
    -------
    .. code-block:: python

        import cuqi
        import numpy as np
        prior = cuqi.distribution.Cauchy_diff(location=np.zeros(128), scale=0.1)

    Notes
    -----
    The pdf is given by

    .. math::

        \pi(\mathbf{x}) = \\frac{1}{(\pi\gamma)^n \left( 1+\left( \\frac{\mathbf{D}(\mathbf{x}-\mathbf{x}_0)}{\gamma} \\right)^2 \\right) },

    where :math:`\mathbf{x}_0\in \mathbb{R}^n` is the location parameter, :math:`\gamma` is the scale, :math:`\mathbf{D}` is the difference operator.
 
    """
   
    def __init__(self, location, scale, bc_type="zero", physical_dim=1, **kwargs):
        # Init from abstract distribution class
        super().__init__(**kwargs) 
        
        self.location = location
        self.scale = scale
        self._bc_type = bc_type
        self._physical_dim = physical_dim

        if physical_dim == 2:
            N = int(np.sqrt(self.dim))
            num_nodes = (N, N)
            if isinstance(self.geometry, _DefaultGeometry):
                self.geometry = Image2D(num_nodes)
            print("Warning: 2D Cauchy_diff is still experimental. Use at own risk.")
        elif physical_dim == 1:
            num_nodes = self.dim
        else:
            raise ValueError("Only physical dimension 1 or 2 supported.")

        self._diff_op = FirstOrderFiniteDifference(num_nodes=num_nodes, bc_type=bc_type)

    @property
    def dim(self): 
        #TODO: handle the case when self.loc = None because len(None) = 1
        return len(self.location)

    def logpdf(self, x):
        Dx = self._diff_op @ (x-self.location)
        # g_logpr = (-2*Dx/(Dx**2 + gamma**2)) @ D
        return -len(Dx)*np.log(np.pi) + sum(np.log(self.scale) - np.log(Dx**2 + self.scale**2))
    
    def gradient(self, val, **kwargs):
        #Avoid complicated geometries that change the gradient.
        if not type(self.geometry) in _get_identity_geometries():
            raise NotImplementedError("Gradient not implemented for distribution {} with geometry {}".format(self,self.geometry))

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
    .. code-block:: python

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
        return np.prod(1/(self.std*np.sqrt(2*np.pi))*np.exp(-0.5*((x-self.mean)/self.std)**2))

    def logpdf(self, x):
        return np.sum(-np.log(self.std*np.sqrt(2*np.pi))-0.5*((x-self.mean)/self.std)**2)

    def cdf(self, x):
        return np.prod(0.5*(1 + erf((x-self.mean)/(self.std*np.sqrt(2)))))

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
        dev = x - self.mean
        mahadist = np.sum(np.square(self.sqrtprec @ dev), axis=-1)
        return -0.5*(self.rank*np.log(2*np.pi) + self.logdet + mahadist)

    def cdf(self, x1):   # TODO
        return sps.multivariate_normal.cdf(x1, self.mean, self.cov)

    def gradient(self, val, *args, **kwargs):
        #Avoid complicated geometries that change the gradient.
        if not type(self.geometry) in _get_identity_geometries():
            raise NotImplementedError("Gradient not implemented for distribution {} with geometry {}".format(self,self.geometry))

        if not callable(self.mean): # for prior
            return -self.prec @ (val - self.mean)
        elif hasattr(self.mean,"gradient"): # for likelihood
            model = self.mean
            dev = val - model.forward(*args, **kwargs)
            return self.prec @ model.gradient(dev)
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

        #Compute permutation
        if N==1: #Ensures we add (dim,1) with (dim,1) and not with (dim,)
            permutation = splinalg.spsolve(self.sqrtprec,e)[:,None]
        else:
            permutation = splinalg.spsolve(self.sqrtprec,e)
            
        # Add to mean
        s = self.mean[:,None] + permutation
        return s


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
        return max(np.size(self.mean),np.shape(self.sqrtprec)[0])

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
            return -self.prec @ (val - self.mean)
        elif hasattr(self.mean,"gradient"): # for likelihood
            model = self.mean
            dev = val - model.forward(*args, **kwargs)
            return self.prec @ model.gradient(dev)
        else:
            warnings.warn('Gradient not implemented for {}'.format(type(self.mean)))

    @property
    def dim(self):
        return max(len(self.mean),self.prec.shape[0])

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


# ========================================================================
class GMRF(Distribution):
    """
        Parameters
        ----------
        mean : array_like
            Mean of the GMRF.

        prec : float
            Precision of the GMRF.

        partition_size : int
            The dimension of the distribution in one physical dimension. 

        physical_dim : int
            The physical dimension of what the distribution represents (can take the values 1 or 2).

        bc_type : str
            The type of boundary conditions to use. Can be 'zero', 'periodic' or 'neumann'.

        order : int
            The order of the GMRF. Can be 1 or 2.
    """
        
    def __init__(self, mean, prec, partition_size, physical_dim, bc_type, order=1, is_symmetric=True, **kwargs): 
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
            if isinstance(self.geometry, _DefaultGeometry):
                self.geometry = Image2D(num_nodes)

        self._prec_op = PrecisionFiniteDifference(num_nodes, bc_type=bc_type, order=order) 
        self._diff_op = self._prec_op._diff_op      
                   
        # compute Cholesky and det
        if (bc_type == 'zero'):    # only for PSD matrices
            self._rank = self.dim
            self._chol = sparse_cholesky(self._prec_op.get_matrix()).T
            self._logdet = 2*sum(np.log(self._chol.diagonal()))
            # L_cholmod = cholesky(self.L, ordering_method='natural')
            # self.chol = L_cholmod
            # self.logdet = L_cholmod.logdet()
            # 
            # np.log(np.linalg.det(self.L.todense()))
        elif (bc_type == 'periodic') or (bc_type == 'neumann'):
            # Print warning that periodic and Neumann boundary conditions are experimental
            print("Warning: Periodic and Neumann boundary conditions are experimental. Sampling using Linear_RTO will not produce fully accurate results.")

            eps = np.finfo(float).eps
            self._rank = self.dim - 1   #np.linalg.matrix_rank(self.L.todense())
            self._chol = sparse_cholesky(self._prec_op + np.sqrt(eps)*eye(self.dim, dtype=int)).T
            if (self.dim > config.MAX_DIM_INV):  # approximate to avoid 'excesive' time
                self._logdet = 2*sum(np.log(self._chol.diagonal()))
            else:
                # eigval = eigvalsh(self.L.todense())
                self._L_eigval = splinalg.eigsh(self._prec_op.get_matrix(), self._rank, which='LM', return_eigenvectors=False)
                self._logdet = sum(np.log(self._L_eigval))
        else:
            raise ValueError('bc_type must be "zero", "periodic" or "neumann"')


    @property 
    def dim(self):  
        if self._physical_dim == 1:
            return self._partition_size 
        elif self._physical_dim==2:
            return self._partition_size**2
        raise ValueError("attribute dom can be either 1 or 2")

    def logpdf(self, x):
        mean = self.mean.flatten()
        const = 0.5*(self._rank*(np.log(self.prec)-np.log(2*np.pi)) + self._logdet)
        y = const - 0.5*( self.prec*((x-mean).T @ (self._prec_op @ (x-mean))) )
        # = sps.multivariate_normal.logpdf(x.T, self.mean.flatten(), np.linalg.inv(self.prec*self.L.todense()))
        return y

    def pdf(self, x):
        # = sps.multivariate_normal.pdf(x.T, self.mean.flatten(), np.linalg.inv(self.prec*self.L.todense()))
        return np.exp(self.logpdf(x))

    def gradient(self, x):
        #Avoid complicated geometries that change the gradient.
        if not type(self.geometry) in _get_identity_geometries():
            raise NotImplementedError("Gradient not implemented for distribution {} with geometry {}".format(self,self.geometry))

        if not callable(self.mean):
            mean = self.mean.flatten()
            return -(self.prec*self._prec_op) @ (x-mean)

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
            
            if self._physical_dim == 2:
                raise NotImplementedError("Sampling not implemented for periodic boundary conditions in 2D")

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
    
    @property
    def sqrtprec(self):
        return np.sqrt(self.prec)*self._chol.T

    @property
    def sqrtprecTimesMean(self):
        return (self.sqrtprec@self.mean).flatten()
        


# ========================================================================
class Laplace_diff(Distribution):
    """Laplace distribution on the difference between neighboring nodes.

    Parameters
    ----------
    location : scalar or ndarray
        The location parameter of the distribution.

    scale : scalar
        The scale parameter of the distribution.

    bc_type : string
        The boundary conditions of the difference operator.

    physical_dim : int
        The physical dimension of what the distribution represents (can take the values 1 or 2).

    Example
    -------
    .. code-block:: python

        import cuqi
        import numpy as np
        prior = cuqi.distribution.Laplace_diff(location=np.zeros(128), scale=0.1)

    Notes
    -----
    The pdf is given by

    .. math::

        \pi(\mathbf{x}) = \\frac{1}{(2b)^n} \exp \left(- \\frac{\|\mathbf{D}(\mathbf{x}-\mathbf{x}_0) \|_1 }{b} \\right),

    where :math:`\mathbf{x}_0\in \mathbb{R}^n` is the location parameter, :math:`b` is the scale, :math:`\mathbf{D}` is the difference operator.
 
    """
    def __init__(self, location, scale, bc_type="zero", physical_dim=1, **kwargs):
        # Init from abstract distribution class
        super().__init__(**kwargs) 

        self.location = location
        self.scale = scale
        self._bc_type = bc_type
        self._physical_dim = physical_dim

        if physical_dim == 2:
            N = int(np.sqrt(self.dim))
            num_nodes = (N, N)
            if isinstance(self.geometry, _DefaultGeometry):
                self.geometry = Image2D(num_nodes)

        elif physical_dim == 1:
            num_nodes = self.dim
        else:
            raise ValueError("Only physical dimension 1 or 2 supported.")

        self._diff_op = FirstOrderFiniteDifference(num_nodes=num_nodes, bc_type=bc_type)


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
        # First check whether x is outside bounds.
        # It is outside if any coordinate is outside the interval.
        if np.any(x < self.low) or np.any(x > self.high):
            # If outside always return -inf
            return_val = -np.inf  
        else:
            # If inside, compute the area and obtain the constant 
            # probability (pdf) as 1 divided by the area, the convert 
            # to logpdf. Special case if scalar.
            diff = self.high - self.low
            if isinstance(diff, (list, tuple, np.ndarray)): 
                v= np.prod(diff)
            else:
                v = diff
            return_val = np.log(1.0/v)
        return return_val

    def _sample(self,N=1, rng=None):

        if rng is not None:
            s = rng.uniform(self.low, self.high, (N,self.dim)).T
        else:
            s = np.random.uniform(self.low, self.high, (N,self.dim)).T

        return s

# ========================================================================
class Posterior(Distribution):
    """
    Posterior probability distribution defined by likelihood and prior.
    The geometry is automatically determined from the model and prior.
    Generates instance of cuqi.distribution.Posterior
    
    Parameters
    ------------
    likelihood: Likelihood function, cuqi.likelihood.Likelihood.
    prior: Prior distribution, cuqi.distribution.Distribution.

    Attributes
    ------------
    likelihood
    prior
    data
    dim
    geometry
    model
    
    Methods
    -----------
    sample: NotImplemented. Use sampler module instead.
    pdf: evaluate probability density function
    logpdf: evaluate log probability density function
    gradient: evaluate the gradient of the log probability density function w.r.t. input parameter.
    """
    def __init__(self, likelihood, prior, **kwargs):
        self.likelihood = likelihood
        self.prior = prior 
        super().__init__(**kwargs)

    @property
    def data(self):
        return self.likelihood.data

    @property
    def dim(self):
        return self.prior.dim

    @property
    def geometry(self):
        return self._geometry

    @geometry.setter
    def geometry(self, value):
        # Compare model and prior
        if self.model is not None and self.model.domain_geometry != self.prior.geometry:
            if isinstance(self.prior.geometry,_DefaultGeometry):
                pass #We allow default geometry in prior
            else:
                raise ValueError("Geometry from likelihood (model.domain_geometry) does not match prior geometry")

        # Compare value and prior
        if self.model is None and value is not None and value != self.prior.geometry:
            if isinstance(self.prior.geometry,_DefaultGeometry):
                pass #We allow default geometry in prior
            else:
                raise ValueError("Posterior and prior geometries are inconsistent.")

        # Compare model and value
        if self.model is not None and value is not None and value != self.model.domain_geometry:
            if isinstance(self.model.domain_geometry,_DefaultGeometry):
                pass #Allow default model geometry
            else:
                raise ValueError("Set geometry does not match with model geometry.")

        # If value is set, its consistant with prior (and prior is consistant with model)
        # If value is not set, take from model (if exists) or from prior as last resort
        if value is not None:
            self._geometry = value
        elif self.model is not None:
            self._geometry = self.model.domain_geometry
        else:
            self._geometry = self.prior.geometry
            
    def logpdf(self, x):
        """ Returns the logpdf of the posterior distribution"""
        return self.likelihood.log(x)+ self.prior.logpdf(x)

    def gradient(self, x):
        #Avoid complicated geometries that change the gradient.
        if not type(self.geometry) in _get_identity_geometries():
            raise NotImplementedError("Gradient not implemented for distribution {} with geometry {}".format(self,self.geometry))
            
        return self.likelihood.gradient(x)+ self.prior.gradient(x)        

    def _sample(self,N=1,rng=None):
        raise Exception("'Posterior.sample' is not defined. Sampling can be performed with the 'sampler' module.")

    @property
    def model(self):
        """Extract the cuqi model from likelihood."""
        return self.likelihood.model

class UserDefinedDistribution(Distribution):
    """
    Class to wrap user-defined logpdf, gradient, and/or sampling callable into CUQIpy Distribution.

    Parameters
    ------------
    logpdf_func: Function evaluating log probability density function. Callable.
    gradient_func: Function evaluating the gradient of the logpdf. Callable.
    logpdf_func: Function evaluating log probability density function. Callable.
    
    Methods
    -----------
    sample: generate one or more random samples
    logpdf: evaluate log probability density function
    gradient: evaluate gradient of logpdf
    
    Example
    -----------
    .. code-block:: python

        # Generate an i.i.d. n-dim Gaussian with zero mean and 2 variance.
        mu1 = -1.0
        std1 = 4.0
        X = cuqi.distribution.Normal(mean=mu1, std=std1)
        dim1 = 1
        logpdf_func = lambda xx: -np.log(std1*np.sqrt(2*np.pi))-0.5*((xx-mu1)/std1)**2
        sample_func = lambda : mu1 + std1*np.random.randn(dim1,1)
        XU = cuqi.distribution.UserDefinedDistribution(dim=dim1, logpdf_func=logpdf_func, sample_func=sample_func)
    """

    def __init__(self, dim=None, logpdf_func=None, gradient_func=None, sample_func=None, **kwargs):

        # Init from abstract distribution class
        super().__init__(**kwargs)

        if logpdf_func is not None and not callable(logpdf_func): raise ValueError("logpdf_func should be callable.")
        if sample_func is not None and not callable(sample_func): raise ValueError("sample_func should be callable.")
        if gradient_func is not None and not callable(gradient_func): raise ValueError("grad_func should be callable.")
        
        self.dim = dim
        self.logpdf_func = logpdf_func
        self.sample_func = sample_func
        self.gradient_func = gradient_func


    @property
    def dim(self):
        return self._dim

    @dim.setter
    def dim(self, value):
        self._dim = value

    def logpdf(self, x):
        if self.logpdf_func is not None:
            return self.logpdf_func(x)
        else:
            raise Exception("logpdf_func is not defined.")
    
    def gradient(self, x):
        if self.gradient_func is not None:
            return self.gradient_func(x)
        else:
            raise Exception("gradient_func is not defined.")

    def _sample(self, N=1, rng=None):
        #TODO(nabr) allow sampling more than 1 sample and potentially rng?
        if self.sample_func is not None:
            if N==1:
                return self.sample_func().flatten()
            else:
                out = np.zeros((self.dim,N))
                for i in range(N):
                    out[:,i] = self.sample_func()
                return out
        else:
            raise Exception("sample_func is not defined. Sampling can be performed with the 'sampler' module.")


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

        super().__init__(logpdf_func=logpdf_func, dim=dim, **kwargs)


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
            s =  rng.laplace(self.location, 1.0/self.prec, (N,self.dim)).T
        else:
            s = np.random.laplace(self.location, 1.0/self.prec, (N,self.dim)).T
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


class Lognormal(Distribution):
    """
    Multivariate Lognormal distribution

    Parameters
    ------------
    mean: np.ndarray
        Mean of the normal distribution used to define the lognormal distribution 

    cov: np.ndarray
        Covariance matrix of the normal distribution used to define the lognormal distribution 
    
    Methods
    -----------
    sample: generate one or more random samples
    pdf: evaluate probability density function
    logpdf: evaluate log probability density function
    cdf: evaluate cumulative probability function
    
    Example
    -------
    .. code-block:: python
    
        # Generate a lognormal distribution
        mean = np.array([1.5,1])
        cov = np.array([[3, 0],[0, 1]])
        x = cuqi.distribution.Lognormal(mean, cov)
        samples = x.sample(10000)
        samples.hist_chain(1, bins=70)

    """
    def __init__(self, mean, cov, is_symmetric=False, **kwargs):
        super().__init__(is_symmetric=is_symmetric, **kwargs) 
        self.mean = mean
        self.cov = cov
        self._normal = GaussianCov(self.mean, self.cov)

    @property
    def _normal(self):
        if not np.all(self._GaussianCov.mean == self.mean):
            self._GaussianCov.mean = self.mean
        if not np.all(self._GaussianCov.cov == self.cov):
            self._GaussianCov.cov = self.cov 
        return self._GaussianCov

    @_normal.setter
    def _normal(self, value):
        self._GaussianCov = value

    @property
    def dim(self):
        return self._normal.dim

    def pdf(self, x):
        if np.any(x<=0):
            return 0
        else:
            return self._normal.pdf(np.log(x))*np.prod(1/x)

    def logpdf(self, x):
        return np.log(self.pdf(x))

    def gradient(self, val, **kwargs):
        #Avoid complicated geometries that change the gradient.
        if not type(self.geometry) in _get_identity_geometries():
            raise NotImplementedError("Gradient not implemented for distribution {} "
                                      "with geometry {}".format(self,self.geometry))

        elif not callable(self._normal.mean): # for prior
            return np.diag(1/val)@(-1+self._normal.gradient(np.log(val)))
        elif hasattr(self.mean,"gradient"): # for likelihood
            model = self._normal.mean
            dev = np.log(val) - model.forward(**kwargs)
            return  model.gradient(self._normal.prec@dev, **kwargs) # Jac(x).T@(self._normal.prec@dev)
        else:
            warnings.warn('Gradient not implemented for {}'.format(type(self._normal.mean)))

    def _sample(self, N=1, rng=None):
        return np.exp(self._normal._sample(N,rng))

class InverseGamma(Distribution):
    """
    Multivariate inverse gamma distribution of independent random variables x_i. Each is distributed according to the PDF function

    f(x) = (x-location)^(-shape-1) * exp(-scale/(x-location)) / (scale^(-shape)*Gamma(shape))

    where shape, location and scale are the shape, location and scale of x_i, respectively. And Gamma is the Gamma function.

    Parameters
    ------------
    shape: float or array_like
        The shape parameter

    location: float or array_like
        The location of the inverse gamma distribution. The support of the pdf function is the Cartesian product of the open intervals (location_1, infinity), (location_2, infinity), ..., (location_dim, infinity).

    scale: float or array_like
        The scale of the inverse gamma distribution (non-negative)

    
    Methods
    -----------
    sample: generate one or more random samples
    pdf: evaluate probability density function
    logpdf: evaluate log probability density function
    cdf: evaluate cumulative probability function
    
    Example
    -------
    .. code-block:: python

        # Generate an InverseGamma distribution
        import numpy as np
        import cuqi
        import matplotlib.pyplot as plt
        shape = [1,2]
        location = 0
        scale = 1
        rng = np.random.RandomState(1)
        x = cuqi.distribution.InverseGamma(shape, location, scale)
        samples = x.sample(1000, rng=rng)
        samples.hist_chain(0, bins=70)
        plt.figure()
        samples.hist_chain(1, bins=70)

    """
    def __init__(self, shape=None, location=None, scale=None, is_symmetric=False, **kwargs):
        super().__init__(is_symmetric=is_symmetric, **kwargs) 
        self.shape = force_ndarray(shape, flatten=True)
        self.location = force_ndarray(location, flatten=True)
        self.scale = force_ndarray(scale, flatten=True)
    
    @property
    def dim(self):
        lens = [ (np.size(item) if item is not None else 0) 
                 for item in [self.shape, self.location, self.scale]]
        return np.max(lens) if np.max(lens)>0 else None

    def logpdf(self, x):
        return np.sum(sps.invgamma.logpdf(x, a=self.shape, loc=self.location, scale=self.scale))

    def cdf(self, x):
        return np.prod(sps.invgamma.cdf(x, a=self.shape, loc=self.location, scale=self.scale))

    def gradient(self, val, **kwargs):
        #Avoid complicated geometries that change the gradient.
        if not type(self.geometry) in _get_identity_geometries():
            raise NotImplementedError("Gradient not implemented for distribution {} with geometry {}".format(self,self.geometry))
        #Computing the gradient for conditional InverseGamma distribution is not supported yet    
        elif self.is_cond:
            raise NotImplementedError(f"Gradient is not implemented for {self} with conditioning variables {self.get_conditioning_variables()}")
        
        #Compute the gradient
        if np.any(val <= self.location):
            return val*np.nan
        else:
            return (-self.shape-1)/(val - self.location) +\
                    self.scale/(val - self.location)**2


    def _sample(self, N=1, rng=None):
        return sps.invgamma.rvs(a=self.shape, loc= self.location, scale = self.scale ,size=(N,self.dim), random_state=rng).T

class Beta(Distribution):
    """
    Multivariate beta distribution of independent random variables x_i. Each is distributed according to the PDF function

    f(x) = x^(alpha-1) * (1-x)^(beta-1) * Gamma(alpha+beta) / (Gamma(alpha)*Gamma(beta))

    where Gamma is the Gamma function.

    Parameters
    ------------
    alpha: float or array_like

    beta: float or array_like

    Methods
    -----------
    sample: generate one or more random samples
    pdf: evaluate probability density function
    logpdf: evaluate log probability density function
    cdf: evaluate cumulative probability function
    gradient: evaluate the gradient of the logpdf
    
    Example
    -------
    .. code-block:: python

        # % Generate a beta distribution
        import numpy as np
        import cuqi
        import matplotlib.pyplot as plt
        alpha = 0.5
        beta  = 0.5
        rng = np.random.RandomState(1)
        x = cuqi.distribution.Beta(alpha, beta)
        samples = x.sample(1000, rng=rng)
        samples.hist_chain(0, bins=70)

    """
    def __init__(self, alpha=None, beta=None, is_symmetric=False, **kwargs):
        super().__init__(is_symmetric=is_symmetric, **kwargs)
        self.alpha = force_ndarray(alpha, flatten=True)
        self.beta = force_ndarray(beta, flatten=True)

    @property
    def dim(self):
        lens = [ (np.size(item) if item is not None else 0) 
                 for item in [self.alpha, self.beta]]
        return np.max(lens) if np.max(lens)>0 else None

    def logpdf(self, x):

        # Check bounds
        if np.any(x<=0) or np.any(x>=1) or np.any(self.alpha<=0) or np.any(self.beta<=0):
            return -np.Inf

        # Compute logpdf
        return np.sum(sps.beta.logpdf(x, a=self.alpha, b=self.beta))

    def cdf(self, x):

        # Check bounds
        if np.any(x<=0) or np.any(x>=1) or np.any(self.alpha<=0) or np.any(self.beta<=0):
            return 0

        # Compute logpdf
        return np.prod(sps.beta.cdf(x, a=self.alpha, b=self.beta))

    def _sample(self, N=1, rng=None):
        return sps.beta.rvs(a=self.alpha, b=self.beta, size=(N,self.dim), random_state=rng).T

    def gradient(self, x):
        #Avoid complicated geometries that change the gradient.
        if not type(self.geometry) in _get_identity_geometries():
            raise NotImplementedError("Gradient not implemented for distribution {} with geometry {}".format(self,self.geometry))
        
        #Computing the gradient for conditional InverseGamma distribution is not supported yet    
        if self.is_cond:
            raise NotImplementedError(f"Gradient is not implemented for {self} with conditioning variables {self.get_conditioning_variables()}")
        
        # Check bounds (return nan if out of bounds)
        if np.any(x<=0) or np.any(x>=1) or np.any(self.alpha<=0) or np.any(self.beta<=0):
            return x*np.nan

        #Compute the gradient
        return (self.alpha - 1)/x + (self.beta-1)/(x-1)
        