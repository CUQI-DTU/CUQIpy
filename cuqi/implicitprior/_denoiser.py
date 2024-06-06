#%%
from abc import ABC, abstractmethod
from cuqi.distribution import Distribution

class DenoiseRegularizer(Distribution):
    """    
    This class defines implicit regularized priors for which we can apply 
    gradient-based algorithms (ex:MYULA). The regularization is performed using
    a denoising algorithm as we can encounter in MYULA
    and PnP-ULA.
    
    Ex: Moreau-Yoshida based regularization
        We consider a density such that - \log\pi(x) = -g(x) with g convex, lsc,
        proper but not differentiable. Consequently, we cannot apply any
        algorithm requiring the gradient of g.
        Idea:
        We consider the Moreau envelope of g defined as
        
        g_{strength_smooth} (x) = inf_z 0.5*\| x-z \|_2^2/strength_smooth + g(z).
        
        g_{strength_smooth} has some nice properties
            - g_{strength_smooth}(x)-->g(x) as strength_smooth-->0 for all x
            - \nabla g_{strength_smooth} is 1/strength_smooth-Lipschitz
            - \nabla g_{strength_smooth}(x) = (x - prox_g^{strength_smooth}(x))/strength_smooth for all x with 
            
            prox_g^{strength_smooth}(x) = argmin_z 0.5*\| x-z \|_2^2/strength_smooth + g(z) .

        Consequently, we can apply any gradient-based algorithm with
        g_{strength_smooth} in lieu of g. These algorithms do not require the
        full knowledge of g_{strength_smooth} but only its gradient.
        It is important as, although there exists an explicit formula for
        g_{strength_smooth}, it is not useable in practice, as it would require
        us to solve an optimization problem each time we want to 
        estimate g_{strength_smooth}.
        The gradient of g_{strength_smooth} is fully determined by
        prox_g^{strength_smooth}.

    Remark (Proximal operators are denoisers):
        We consider the denoising inverse problem x = u + n, with
        n \sim \mathcal{N}(0, strength_smooth I).
        We assume a prior density \pi(u) \propto exp(- g(u)).
        Then the MAP estimate is given by 
            x_MAP = \argmin_z 0.5 \| x - z \|_2^2/strength_smooth + g(z) = prox_g^strength_smooth(x) ()
        Then proximal operators are denoisers. 
    
    Remark (Denoisers are not necessarily proximal operators): Data-driven
    denoisers are not necessarily proximal operators
    (see https://arxiv.org/pdf/2201.13256)
    
    Parameters
    ---------- 
    denoiser : callable f(x)
        Denoising function f that accepts input x to be denoised and returns the
        denoised version of x and information about the denoising algorithm in a
        dictionary such as the number of iterations, the accuracy, eg
        {"num_itr": 100, "accuracy":  0.1}
        
    denoiser_kwargs : dictionary
        Dictionary containing information such as the denoising strength or the
        prior regularization strength
        
    strength_smooth : float
        Smoothing strength
    """

    def __init__(self, denoiser, denoiser_kwargs = None, strength_smooth = 0.1, 
                 **kwargs):
        if denoiser_kwargs is None:
            denoiser_kwargs = {}
        self.denoiser = denoiser
        self.denoiser_kwargs = denoiser_kwargs
        self.strength_smooth = strength_smooth
        super().__init__(**kwargs)

    def regularize(self, x):
        """This function allows us to denoise the input x and returns the
        denoised version of x.
        """
        solution, info = self.denoiser(x, **self.denoiser_kwargs)
        self.info = info
        return solution
    
    def gradient(self, x):
        """This is the gradient of the regularizer."""
        return -(x - self.regularize(x))/self.strength_smooth
    
    def logpdf(self, x):
        raise NotImplementedError("The logpdf method is not implemented for" + 
                                  "the DenoiseRegularizer class.")
    
    def _sample(self, N, rng=None):
        raise NotImplementedError("The sample method is not implemented for the"
                                  + "DenoiseRegularizer class.")

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