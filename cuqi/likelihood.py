#from cuqi.distribution import Distribution #How to avoid circular import?
from cuqi.model import Model
from cuqi.utilities import getNonDefaultArgs
import warnings

class Likelihood(object):
    """Likelihood function defined from a conditional distribution and observation.

    The likelihood takes the value of the logpdf of a distribution given an observation.
    The geometry is automatically determined from the model of data distribution.
    Generates instance of cuqi.likelihood.Likelihood
    
    Parameters
    ------------
    distribution: distribution to create likelihood from.
    data: observation to create likelihood from.

    Attributes
    ------------
    distribution
    data
    dim
    shape
    geometry
    model
    
    Methods
    -----------
    log: evaluate log density function w.r.t input parameter.
    gradient: evaluate the gradient of the log density function w.r.t. input parameter.
    get_parameter_names: gets the input parameter names.
    """

    def __init__(self, distribution, data):
        # Check if distribution is conditional
        if not distribution.is_cond:
            raise TypeError("Input distribution must be a conditional distribution to convert to likelihood.")
        
        self.distribution = distribution
        self.data = data

        # Check if a CUQI model is inside distribution
        self.model # returns error if distribution does not have model

    def log(self, *args, **kwargs):
        """Return the log-likelihood function at given value"""
        return self.distribution(*args, **kwargs).logpdf(self.data)

    def gradient(self, *args, **kwargs):
        """Return gradient of the log-likelihood function at given value"""
        return self.distribution.gradient(self.data, *args, **kwargs)

    @property
    def dim(self):
        if len(self.get_parameter_names()) > 1:
            warnings.warn("returned dim is only w.r.t. parameter of model input, but likelihood has more parameters!")
        return self.model.domain_dim

    @property
    def shape(self):
        if len(self.get_parameter_names()) > 1:
            warnings.warn("returned shape is only w.r.t. parameter of model input, but likelihood has more parameters!")
        return self.model.domain_geometry.shape

    @property
    def geometry(self):
        if len(self.get_parameter_names()) > 1:
            warnings.warn("returned geometry is only w.r.t. parameter of model input, but likelihood has more parameters!")
        return self.model.domain_geometry

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
    """Class to wrap user-defined likelihood functions.

    Parameters
    ------------
    dim: Dimension of the likelihood. Int.
    logpdf_func: Function evaluating log density function. Callable.
    gradient_func: Function evaluating the gradient of the log density. Callable.
    geometry: cuqi.geometry.Geometry.
    
    Methods
    -----------
    log: evaluate log density function w.r.t input parameter.
    gradient: evaluate the gradient of the log density function w.r.t. input parameter.
    get_parameter_names: gets the input parameter names.
    """

    def __init__(self, dim=None, logpdf_func=None, gradient_func=None, geometry=None):
        self.dim = dim
        self.logpdf_func = logpdf_func
        self.gradient_func = gradient_func
        self.geometry = geometry

    @property
    def dim(self):
        return self._dim

    @dim.setter
    def dim(self, value):
        self._dim = value

    def log(self, *args, **kwargs):
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
