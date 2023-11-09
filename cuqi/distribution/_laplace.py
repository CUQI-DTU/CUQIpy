import numpy as np
from cuqi.distribution import Distribution

class Laplace(Distribution):
    """ Laplace distribution. 

    Defines a Laplace distribution given a location and a scale. The Laplace distribution is defined as

    .. math::

        p(x) = \\frac{1}{2b} \exp\left(-\\frac{|x-\mu|}{b}\\right),

    where :math:`\mu` is the location (mean) and :math:`b` is the scale (decay) parameter.

    The rate parameter is defined as :math:`\lambda = \\frac{1}{b}`.

    The variables of this Laplace distribution are independent identically distributed (i.i.d.).

    Parameters
    ----------
    location : scalar or ndarray
        The location parameter of the distribution.

    scale : scalar
        The scale parameter of the distribution.

    """

    def __init__(self, location, scale, **kwargs):

        # Init from abstract distribution class
        super().__init__(**kwargs)

        self.location = location
        self.scale = scale
        self.rate = 1.0/self.scale
  
    def logpdf(self, x):
        if isinstance(x, (float,int)):
            x = np.array([x])
        return self.dim*(np.log(self.rate/2)) - self.rate*np.linalg.norm(x-self.location,1)

    def _sample(self,N=1,rng=None):
        if rng is not None:
            s =  rng.laplace(self.location, 1.0/self.rate, (N,self.dim)).T
        else:
            s = np.random.laplace(self.location, 1.0/self.rate, (N,self.dim)).T
        return s
