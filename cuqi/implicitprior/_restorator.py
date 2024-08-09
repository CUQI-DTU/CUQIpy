#%%
from abc import ABC, abstractmethod
from cuqi.distribution import Distribution
import numpy as np
    
class RestorationPrior(Distribution):
    """    
    This class defines an implicit distribution associated with a restoration operator
    (eg denoiser). They are several works relating restorations operators with
    priors, see 
        -Laumont et al. https://arxiv.org/abs/2103.04715
        -Hu et al. https://openreview.net/pdf?id=x7d1qXEn1e
    We cannot sample from this distribution, neither compute its logpdf. It allows
    us to apply algorithms such as MYULA and PnPULA.
    
    Parameters
    ---------- 
    restorator : callable f(x, restoration_strength)
        Function f that accepts input x to be restored and returns the
        restored version of x and information about the restoration operation.
        
    restorator_strength : float
        Strength of the restoration operation. In the case where the restorator is
        a denoiser, this parameter might correspond to the noise level.
        
    restorator_kwargs : dictionary
        Dictionary containing information about the restorator.
    
    potential : callable function
        In the case, where the restoration operator is the proximal operator of
        the function potential. This function is a mapping from the image domain
        to the real set. 
    """

    def __init__(self, restorator, restoration_strength=None, restorator_kwargs
                =None, potential=None, **kwargs):
        if restorator_kwargs is None:
            restorator_kwargs = {}
        self.restorator = restorator 
        self.restorator_kwargs = restorator_kwargs
        self.restoration_strength = restoration_strength
        self.potential = potential
        super().__init__(**kwargs)

    def restorate(self, x):
        """This function allows us to denoise the input x and returns the
        denoised version of x.
        """
        solution, info = self.restorator(x, restoration_strength=self.restoration_strength,
                                         **self.restorator_kwargs)
        self.info = info
        return solution
    
    def logpdf(self, x):
        """The logpdf function. It returns nan because we don't know the 
        logpdf of the implicit prior."""
        if self.potential == None:
            return np.nan
        else:
            return -self.potential(x)
    
    def _sample(self, N, rng=None):
        raise NotImplementedError("The sample method is not implemented for the"
                                  + "RestorationPrior class.")

    #TODO this method copied from userdefinedistribution
    @property
    def _mutable_vars(self):
        """ Returns the mutable variables of the distribution. """
        # Currently mutable variables are not supported for user-defined
        # distributions.
        return []

    #TODO this method copied from userdefinedistribution
    def get_conditioning_variables(self):
        """ Returns the conditioning variables of the distribution. """
        # Currently conditioning variables are not supported for user-defined
        # distributions.
        return []
############################################################################################   
class MoreauYoshidaPrior(Distribution):
    """    
    This class defines (implicit) smoothed priors for which we can apply 
    gradient-based algorithms. The smoothing is performed using
    the Moreau-Yoshida envelope of the target prior potential.
    
    In the following we give a detailed explanation of the
    Moreau-Yoshida smoothing.
    
    We consider a density such that - \log\pi(x) = -g(x) with g convex, lsc,
    proper but not differentiable. Consequently, we cannot apply any
    algorithm requiring the gradient of g.
    Idea:
    We consider the Moreau envelope of g defined as
    
    g_{smoothing_strength} (x) = inf_z 0.5*\| x-z \|_2^2/smoothing_strength + g(z).
    
    g_{smoothing_strength} has some nice properties
        - g_{smoothing_strength}(x)-->g(x) as smoothing_strength-->0 for all x
        - \nabla g_{smoothing_strength} is 1/smoothing_strength-Lipschitz
        - \nabla g_{smoothing_strength}(x) = (x - prox_g^{smoothing_strength}(x))/smoothing_strength for all x with 
        
        prox_g^{smoothing_strength}(x) = argmin_z 0.5*\| x-z \|_2^2/smoothing_strength + g(z) .

    Consequently, we can apply any gradient-based algorithm with
    g_{smoothing_strength} in lieu of g. These algorithms do not require the
    full knowledge of g_{smoothing_strength} but only its gradient. The gradient
    of g_{smoothing_strength} is fully determined by prox_g^{smoothing_strength}
    and smoothing_strength.
    It is important as, although there exists an explicit formula for
    g_{smoothing_strength}, it is rarely used in practice, as it would require
    us to solve an optimization problem each time we want to 
    estimate g_{smoothing_strength}. Furthermore, there exist cases where we dont't
    the regularization g with which the mapping prox_g^{smoothing_strength} is
    associated.

    Remark (Proximal operators are denoisers):
        We consider the denoising inverse problem x = u + n, with
        n \sim \mathcal{N}(0, smoothing_strength I).
        We assume a prior density \pi(u) \propto exp(- g(u)).
        Then the MAP estimate is given by 
            x_MAP = \argmin_z 0.5 \| x - z \|_2^2/smoothing_strength + g(z) = prox_g^smoothing_strength(x) ()
        Then proximal operators are denoisers. 
    
    Remark (Denoisers are not necessarily proximal operators): Data-driven
    denoisers are not necessarily proximal operators
    (see https://arxiv.org/pdf/2201.13256)
    
    Parameters
    ---------- 
    prior : RestorationPrior
        Prior of the Restoration Prior type. In order to stay within the MYULA
        framework the restorator of RestorationPrior must be a proximal operator.
        
    smoothing_strength : float
        Smoothing strength of the Moreau-Yoshida envelope of the prior potential.
        smoothing_strength needs to be equal to restoration_strength of the prior
        ResotrationPrior.
    """

    def __init__(self, prior:RestorationPrior, smoothing_strength=0.1, 
                 **kwargs):
        self.prior = prior
        self.smoothing_strength = smoothing_strength
        super().__init__(**kwargs)
        
    @property
    def smoothing_strength(self):
        """ smoothing_strength of the distribution"""
        return self._smoothing_strength

    @smoothing_strength.setter
    def smoothing_strength(self, value):
        if self.prior.restoration_strength is not None and not \
            np.isclose(value, self.prior.restoration_strength):
            raise ValueError(f"smoothing_strength of the {self.__class__.__name__}"
                             +"must be equal to restoration_strength of the" 
                             + f"{self.prior.__class__.__name__}.")
        self._smoothing_strength = value
        
    @property
    def prior(self):
        """Getter for the MoreauYoshida prior."""
        return self._prior

    @prior.setter
    def prior(self, value):
        self._prior = value
    
    def gradient(self, x):
        """This is the gradient of the regularizer ie gradient of the negative
        logpdf of the implicit prior."""
        return -(x - self.prior.restorate(x))/self.smoothing_strength
        
    def logpdf(self, x):
        """The logpdf function. It returns nan because we don't know the 
        logpdf of the implicit prior."""
        if self.prior.potential == None:
            return np.nan
        else:
            return -(self.prior.potential(self.prior.restorate(x))*self.smoothing_strength +
                     0.5*((x-self.prior.restorate(x))**2).sum())
    
    def _sample(self, N, rng=None):
        raise NotImplementedError("The sample method is not implemented for the"
                                  + f"{self.__class__.__name__} class.")

    #TODO this method copied from userdefinedistribution
    @property
    def _mutable_vars(self):
        """ Returns the mutable variables of the distribution. """
        # Currently mutable variables are not supported for user-defined
        # distributions.
        return []

    #TODO this method copied from userdefinedistribution
    def get_conditioning_variables(self):
        """ Returns the conditioning variables of the distribution. """
        # Currently conditioning variables are not supported for user-defined
        # distributions.
        return []