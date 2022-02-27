#from cuqi.distribution import Distribution #How to avoid circular import?
from cuqi.model import Model
from cuqi.utilities import getNonDefaultArgs

class Likelihood(object):
    """Likelihood function"""

    def __init__(self, distribution, data):
        if not distribution.is_cond:
            raise TypeError("Input distribution must be a conditional distribution to convert to likelihood.")
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

    @property
    def model(self):
        """Extract the cuqi model from data distribution."""

        model_value = None

        for key, value in vars(self.distribution).items():
            if isinstance(value,Model):
                if model_value is None:
                    model_value = value
                else:
                    raise ValueError("Multiple cuqi models found in dist. This is not supported at the moment.")
        
        if model_value is None:
            #If no model was found we also give error
            raise TypeError("Cuqi model could not be extracted from distribution {}".format(self.distribution))
        else:
            return model_value

class UserDefinedLikelihood(object):
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
