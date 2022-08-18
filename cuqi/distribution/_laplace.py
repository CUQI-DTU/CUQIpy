import numpy as np
from cuqi.distribution import Distribution

class Laplace(Distribution):
    """
    The variables of this Laplace distribution are iid.
    """

    def __init__(self, location, prec, **kwargs):

        # Init from abstract distribution class
        super().__init__(**kwargs)

        self.location = location
        self.prec = prec
  
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
