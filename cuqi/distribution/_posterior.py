from cuqi.geometry import _DefaultGeometry, _get_identity_geometries
from cuqi.distribution import Distribution
from cuqi.likelihood import Likelihood
from abc import ABC, abstractmethod
from cuqi.implicitprior import Regularizer
# ========================================================================
class _GenericPosterior(Distribution, ABC):
    """
    Generic posterior class to represent classical Bayesian posterior 
    distribution and implicitly defined posterior distributions.
    
    Parameters
    ------------
    kwargs: dict
        Parameters to be passed to the Distribution class.

    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    @abstractmethod
    def data(self):
        pass

    @property
    @abstractmethod
    def dim(self):
        pass

    @property
    @abstractmethod
    def geometry(self):
        pass

    @property
    @abstractmethod
    def model(self):
        pass

    @abstractmethod
    def logpdf(self, *args, **kwargs):
        """ Returns the logpdf of the generic posterior distribution"""
        pass

    @abstractmethod
    def get_conditioning_variables(self):
        pass

    @abstractmethod
    def get_parameter_names(self):
        pass

    @abstractmethod
    def _gradient(self, x):
        pass

    def _sample(self, N=1, rng=None):
        raise Exception("'Posterior.sample' is not defined. Sampling can be performed with the 'sampler' module.")

    def __repr__(self):
        msg = "_GenericPosterior()"
        return msg

class Posterior(_GenericPosterior):
    """
    Posterior probability distribution defined by likelihood and prior.
    The geometry is automatically determined from the model and prior.
    Generates instance of cuqi.distribution.Posterior
    
    Parameters
    ------------
    likelihood: Likelihood function, cuqi.likelihood.Likelihood.
    prior: Prior distribution, cuqi.distribution.Distribution.

    """
    def __init__(self, likelihood, prior, **kwargs):

        if len(likelihood.get_parameter_names()) > 1:
            raise ValueError("Likelihood must only have one parameter.")
        if prior.is_cond:
            raise ValueError("Prior must not be a conditional distribution.")

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
            if isinstance(self.prior.geometry, _DefaultGeometry):
                pass #We allow default geometry in prior
            else:
                raise ValueError("Geometry from likelihood (model.domain_geometry) does not match prior geometry")

        # Compare value and prior
        if self.model is None and value is not None and value != self.prior.geometry:
            if isinstance(self.prior.geometry, _DefaultGeometry):
                pass #We allow default geometry in prior
            else:
                raise ValueError("Posterior and prior geometries are inconsistent.")

        # Compare model and value
        if self.model is not None and value is not None and value != self.model.domain_geometry:
            if isinstance(self.model.domain_geometry, _DefaultGeometry):
                pass #Allow default model geometry
            else:
                raise ValueError("Set geometry does not match with model geometry.")
            
        # Compare likelihood and prior
        if self.likelihood.geometry != self.prior.geometry:
            if isinstance(self.prior.geometry, _DefaultGeometry):
                pass #We allow default geometry in prior
            elif isinstance(self.likelihood.geometry, _DefaultGeometry):
                pass #We allow default geometry in likelihood
            else:
                raise ValueError("Likelihood and prior geometries are inconsistent.")
        
        # If value is set, its consistant with prior (and prior is consistant with model)
        # Likelihood and prior are consistant.
        # If value is not set, take from model (if exists) or from likelihood or prior as last resort
        if value is not None and not isinstance(value, _DefaultGeometry):
            self._geometry = value
        elif self.model is not None and not isinstance(self.model.domain_geometry, _DefaultGeometry):
            self._geometry = self.model.domain_geometry
        elif not isinstance(self.likelihood.geometry, _DefaultGeometry):
            self._geometry = self.likelihood.geometry
        else:
            self._geometry = self.prior.geometry
            
    def logpdf(self, *args, **kwargs):
        """ Returns the logpdf of the posterior distribution"""
        return self.likelihood.logd(*args, **kwargs)+ self.prior.logd(*args, **kwargs)

    def get_conditioning_variables(self):
        return self.prior.get_conditioning_variables()

    def get_parameter_names(self):
        return self.prior.get_parameter_names()

    def _gradient(self, x):
        #Avoid complicated geometries that change the gradient.
        if not type(self.geometry) in _get_identity_geometries() and\
           not hasattr(self.geometry, 'gradient'):
            raise NotImplementedError("Gradient not implemented for distribution {} with geometry {}".format(self,self.geometry))
            
        return self.likelihood.gradient(x)+ self.prior.gradient(x)        

    def _sample(self,N=1,rng=None):
        raise Exception("'Posterior.sample' is not defined. Sampling can be performed with the 'sampler' module.")

    @property
    def model(self):
        """Extract the cuqi model from likelihood."""
        return self.likelihood.model

    def __repr__(self):
        msg = f"Posterior(\n"
        msg += "    Equation:\n"
        msg += f"\t p({self.prior.name}|{self.likelihood.name}) ∝ L({self.prior.name}|{self.likelihood.name})p({self.prior.name})\n"
        msg += "    Densities:\n"
        msg += f"\t{self.likelihood.name} ~ {self.likelihood}\n "
        msg += f"\t{self.prior.name} ~ {self.prior}\n "
        msg += ")"
        return msg

class ImplicitlyDefinedPosterior(_GenericPosterior):
    """
    A class representing sampling target distribution that are implicitly
    defined and approximates the Bayesian Posterior. The target distribution
    is defined by a combination of all or some of the following:
    - Likelihood function
    - Prior distribution
    - Regularization term (including denoising, sparsity, etc.)
    
    Parameters
    ------------
    functionals: list of cuqi.functional.Functional #TODO implement functional
        class

    """
    def __init__(self, *functionals, **kwargs):
        self.functionals = functionals
        super().__init__(**kwargs)

    @property
    def functionals(self):
        return [self.likelihood, self.prior] + self.regularizers
    
    @functionals.setter
    def functionals(self, value):
        # Checks for Likelihood:
        # 1. Assert at most one likelihood
        likelihoods = [f for f in value if isinstance(f, Likelihood)]
        if len(likelihoods) > 1:
            raise ValueError("At most one likelihood is allowed.")
        # 2. Assert likelihood has only one parameter
        if len(likelihoods[0].get_parameter_names()) > 1:
            raise ValueError("Likelihood must only have one parameter.")
        self.likelihood = likelihoods[0] if len(likelihoods) > 0 else None

        # Checks for Prior:
        # 1. Assert at most one prior
        priors = [f for f in value if isinstance(f, Distribution)]
        if len(priors) > 1:
            raise ValueError("At most one prior is allowed.")
        # 2. Assert prior is not conditional
        if len(priors) > 0 and priors[0].is_cond:
            raise ValueError("Prior must not be a conditional distribution.")
        self.prior = priors[0] if len(priors) > 0 else None

        # Checks for Regularization:
        # Assert all the remaining functionals are regularization terms
        regularizers = \
            [f for f in value if f not in likelihoods and f not in priors]
        assert all([isinstance(f, Regularizer) for f in regularizers]),\
        "All the remaining functionals must be regularization terms."
        self.regularizers = regularizers

        # Assert all functionals are either likelihood, prior or regularization
        assert len(likelihoods) + len(priors) + len(regularizers) == len(value),\
        "All functionals must be either likelihood, prior or regularization terms."

    @property
    def data(self):
        if self.likelihood is not None:
            return self.likelihood.data
        else:
            raise ValueError(
                "Data is not defined for implicitly defined posterior distribution.")

    @property
    def dim(self):
        # dim from prior
        if self.prior is not None:
            return self.prior.dim
        # dim from likelihood
        elif self.likelihood is not None:
            return self.likelihood.dim
        else:
            raise ValueError(
                "Dimension is not defined for implicitly defined posterior distribution.")

    @property
    def geometry(self):
        # geometry from prior
        if self.prior is not None:
            return self.prior.geometry
        # geometry from model
        elif self.model is not None:
            return self.model.domain_geometry
        else:
            raise ValueError(
                "Geometry is not defined for implicitly defined posterior distribution.")

    @property
    def model(self):
        return None

    def logpdf(self, *args, **kwargs):
        """ Returns the logpdf of the implicitly defined posterior distribution"""
        return sum([f(*args, **kwargs) for f in self.functionals])
        #TODO: should logpdf be implemented for functionals, or just eval?

    def get_conditioning_variables(self):
        if self.prior is not None:
            return self.prior.get_conditioning_variables()
        return None

    def get_parameter_names(self):
        if self.prior is not None:
            return self.prior.get_parameter_names()
        return None

    def _gradient(self, x):
        return sum([f.gradient(x) for f in self.functionals])

    def _sample(self, N=1, rng=None):
        raise Exception("'ImplicitlyDefinedPosterior.sample' is not defined. Sampling can be performed with the 'sampler' module.")

    def __repr__(self):
        msg = f"ImplicitlyDefinedPosterior(\n"
        msg += "    Functionals:\n"
        for f in self.functionals:
            msg += f"\t{f}\n"
        msg += ")"
        return msg