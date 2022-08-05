""" 
The :class:`~cuqi.likelihood.Likelihood` represents the likelihood function commonly used in conjunction
with a prior to define a :class:`cuqi.distribution.Posterior` or :class:`cuqi.problem.BayesianProblem`.

Quick example
-------------
Create a Gaussian likelihood function from a forward `model` and observed `data`

.. code-block:: python

   import cuqi
   model, data, probInfo = cuqi.testproblem.Deconvolution1D.get_components()
   likelihood = cuqi.distribution.Gaussian(mean=model, std=0.05).to_likelihood(data)


Mathematical details
--------------------

Given a conditional distribution :math:`\pi(b \mid x)` and a observed data :math:`b^{obs}` the likelihood function is defined as

.. math::
   
   L(x \mid b^{obs}): x \to \pi(b^{obs} \mid x).

The most commonly use-case of the likelihood function is to define the likelihood function for a Bayesian inverse problem with Gaussian measurement noise.

Consider the inverse problem

.. math::

   b^{obs} = A(x)+e,

where :math:`b^{obs}\in\mathbb{R}^m` is the (noisy) measurement, :math:`e\in\mathbb{R}^m` is the measurement noise, :math:`x\in\mathbb{R}^n` is the solution to the inverse problem and :math:`A: \mathbb{R}^n \to \mathbb{R}^m` is the forward model.

The stochastic extension of the inverse problem is

.. math::
   B = AX+E,

where :math:`B, X` and :math:`E` are random variables.

Assuming Gaussian measurement noise :math:`E\sim\mathcal{N}(0, \sigma^2 I)` the data follows the distribution :math:`B \mid X=x \sim\mathcal{N}(A(x),\sigma^2I)` and given an observation :math:`b^{obs}` the likelihood function is given by

.. math::

   L(x \mid b^{obs}) \propto \exp\left( -\\frac{1}{2\sigma^2} \|b^{obs}-Ax\|_2^2 \\right).

"""

from cuqi.model import Model
from cuqi.utilities import get_non_default_args
from cuqi.geometry import _DefaultGeometry
import warnings
from copy import copy

class Likelihood(object):
    """Likelihood function defined from a conditional distribution and some observed data.

    The parameters of the likelihood function is defined as the conditioning variables
    of a conditional distribution.

    The geometry is automatically determined from the model of data distribution.
    Generates instance of cuqi.likelihood.Likelihood
    
    Parameters
    ------------
    distribution : ~cuqi.core.Distribution
        | Distribution to create likelihood from.
    
    data : ~cuqi.samples.CUQIarray or array_like
        | Observation to create likelihood from.

    """

    def __init__(self, distribution, data):
        # Check if distribution is conditional
        if not distribution.is_cond:
            raise TypeError("Input distribution must be a conditional distribution to convert to likelihood.")
        
        self.distribution = distribution
        self.data = data

    def log(self, *args, **kwargs):
        """Return the log-likelihood function at given value"""
        return self.distribution(*args, **kwargs).logpdf(self.data)

    def gradient(self, *args, **kwargs):
        """Return gradient of the log-likelihood function at given value"""
        return self.distribution.gradient(self.data, *args, **kwargs)

    @property
    def dim(self):
        """ Return dimension of likelihood """
        return self.geometry.dim

    @property
    def shape(self):
        """ Return shape of likelihood """
        return self.geometry.shape

    @property
    def geometry(self):
        """ Return geometry of likelihood """
        if self.model is None:
            return _DefaultGeometry()
        if len(self.get_parameter_names()) > 1:
            warnings.warn(
                f"Likelihood depends on multiple parameters {self.get_parameter_names()}.\n"
                f"Returned geometry is only with respect to the model parameter {get_non_default_args(self.model)}."
            )
        return self.model.domain_geometry

    def get_parameter_names(self):
        """Return parameter names of likelihood"""
        return self.distribution.get_conditioning_variables()

    def __repr__(self) -> str:
        return "CUQI {} {} function. Parameters {}.".format(self.distribution.__class__.__name__,self.__class__.__name__,self.get_parameter_names())

    @property
    def model(self) -> Model:
        """ Extract model from data distribution.
        
        Returns
        -------
        model: cuqi.model.Model or None
            Forward model used in defining the data distribution or None if no model is found.

        """

        model_value = None

        for var in self.distribution.get_mutable_variables():
            value = getattr(self.distribution, var)
            if isinstance(value, Model):
                if model_value is None:
                    model_value = value
                else:
                    raise ValueError(f"Multiple models found in data distribution {self.distribution} of {self}. Extracting model is ambiguous and not supported.")
        
        return model_value

    def __call__(self, *args, **kwargs):
        """ Fix some parameters of the likelihood function by conditioning on the underlying distribution. """
        new_likelihood = copy(self)
        new_likelihood.distribution = self.distribution(*args, **kwargs)
        return new_likelihood

class UserDefinedLikelihood(object):
    """ Class to wrap user-defined likelihood functions.

    Parameters
    ------------
    dim : int 
        Dimension of the likelihood.

    logpdf_func : callable
        Function evaluating log density function.

    gradient_func : callable
        Function evaluating the gradient of the log density.

    geometry : Geometry
        Geometry of the likelihood.
    
    """

    def __init__(self, dim=None, logpdf_func=None, gradient_func=None, geometry=None):
        self.dim = dim
        self.logpdf_func = logpdf_func
        self.gradient_func = gradient_func
        self.geometry = geometry

    @property
    def dim(self):
        """ Return dimension of likelihood """
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
        return get_non_default_args(self.logpdf_func)

    def __repr__(self) -> str:
        return "CUQI {} function. Parameters {}.".format(self.__class__.__name__,self.get_parameter_names())
