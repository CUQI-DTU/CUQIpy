from cuqi.distribution import Distribution
from cuqi.utilities import getNonDefaultArgs

class Likelihood(object):
    """Likelihood function"""

    def __init__(self, distribution: Distribution, data):
        self.distribution = distribution
        self.data = data

    def __call__(self, *args, **kwargs):
        """Returns value of likelihood function"""
        return self.distribution(*args, **kwargs).logpdf(self.data)

    def gradient(self, *args, **kwargs):
        """Return gradient of likelihood function"""
        return self.distribution.gradient(self.data, *args, **kwargs)

    @property
    def dim(self):
        raise NotImplementedError("Dimension is not yet implemented.")

    def get_parameter_names(self):
        """Return parameter names of likelihood"""
        return self.distribution.get_conditioning_variables()

    def __repr__(self) -> str:
        return "CUQI {} {} function. Parameters {}.".format(self.distribution.__class__.__name__,self.__class__.__name__,self.get_parameter_names())

class UserDefinedLikelihood(Likelihood):
    """User defined likelihood"""

    def __init__(self, dim, logpdf_func, gradient_func=None):
        self.dim = dim
        self.logpdf_func = logpdf_func
        self.gradient_func = gradient_func

    @property
    def dim(self):
        return self._dim

    @dim.setter
    def dim(self, value):
        self._dim = value

    def __call__(self, *args, **kwargs):
        """Returns value of likelihood function"""
        return self.logpdf_func(*args, **kwargs)

    def gradient(self, *args, **kwargs):
        """Return gradient of likelihood function"""
        return self.gradient_func(*args, **kwargs)

    def get_parameter_names(self):
        """Return parameter names of likelihood"""
        return getNonDefaultArgs(self.logpdf_func)

    def __repr__(self) -> str:
        return "CUQI {} function. Parameters {}.".format(self.__class__.__name__,self.get_parameter_names())
