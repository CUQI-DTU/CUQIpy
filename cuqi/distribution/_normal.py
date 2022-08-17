import numpy as np
from scipy.special import erf
from cuqi.distribution import Distribution

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