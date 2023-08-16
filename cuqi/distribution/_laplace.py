import numpy as np
from cuqi.distribution import Distribution
from cuqi.utilities import to_cuqi_format

class Laplace(Distribution):
    """ Laplace distribution. 

    Defines a Laplace distribution given a location and a precision. The Laplace distribution is defined as

    .. math::

        p(x) = \\frac{1}{2b} \exp\left(-\\frac{|x-\mu|}{b}\\right),

    where :math:`\mu` is the location and :math:`b` is the scale parameter.

    The precision is defined as :math:`\lambda = \\frac{1}{b}`.

    The variables of this Laplace distribution are independent identically distributed (i.i.d.).

    Parameters
    ----------
    location : scalar or ndarray
        The location parameter of the distribution.

    prec : scalar
        The precision parameter of the distribution.

    """

    def __init__(self, location, prec, **kwargs):

        # Init from abstract distribution class
        super().__init__(**kwargs)

        self.location = location
        self.prec = prec

    @property
    def location(self):
        return self._location
    
    @location.setter
    def location(self, value):
        self._location = to_cuqi_format(value, force_ndarray=False)

    @property
    def prec(self):
        return self._prec
    
    @prec.setter
    def prec(self, value):
        self._prec = to_cuqi_format(value, force_ndarray=False)
  
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
