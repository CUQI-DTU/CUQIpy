from cuqi.geometry import _DefaultGeometry, _get_identity_geometries
from cuqi.distribution import Distribution

# ========================================================================
class Posterior(Distribution):
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
