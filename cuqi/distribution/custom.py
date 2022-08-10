import numpy as np
from cuqi.core import Distribution
from cuqi.distribution import Gaussian

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
