import numpy as np
import cuqi
from cuqi.experimental.mcmc import Sampler
from cuqi.implicitprior import RestorationPrior, MoreauYoshidaPrior
from cuqi.array import CUQIarray
from copy import copy

class ULA(Sampler): # Refactor to Proposal-based sampler?
    """Unadjusted Langevin algorithm (ULA) (Roberts and Tweedie, 1996)

    It approximately samples a distribution given its logpdf gradient based on
    the Langevin diffusion dL_t = dW_t + 1/2*Nabla target.logd(L_t)dt, where
    W_t is the `dim`-dimensional standard Brownian motion.
    ULA results from the Euler-Maruyama discretization of this Langevin stochastic
    differential equation (SDE). 

    For more details see: Roberts, G. O., & Tweedie, R. L. (1996). Exponential convergence
    of Langevin distributions and their discrete approximations. Bernoulli, 341-363.

    Parameters
    ----------

    target : `cuqi.distribution.Distribution`
        The target distribution to sample. Must have logd and gradient method. Custom logpdfs 
        and gradients are supported by using a :class:`cuqi.distribution.UserDefinedDistribution`.
    
    initial_point : ndarray
        Initial parameters. *Optional*

    scale : float
        The Langevin diffusion discretization time step (In practice, scale must
        be smaller than 1/L, where L is the Lipschitz of the gradient of the log
        target density, logd).

    callback : callable, optional
        A function that will be called after each sampling step. It can be useful for monitoring the sampler during sampling.
        The function should take three arguments: the sampler object, the index of the current sampling step, the total number of requested samples. The last two arguments are integers. An example of the callback function signature is: `callback(sampler, sample_index, num_of_samples)`.


    Example
    -------
    .. code-block:: python

        # Parameters
        dim = 5 # Dimension of distribution
        mu = np.arange(dim) # Mean of Gaussian
        std = 1 # standard deviation of Gaussian

        # Logpdf function
        logpdf_func = lambda x: -1/(std**2)*np.sum((x-mu)**2)
        gradient_func = lambda x: -2/(std**2)*(x - mu)

        # Define distribution from logpdf and gradient as UserDefinedDistribution
        target = cuqi.distribution.UserDefinedDistribution(dim=dim, logpdf_func=logpdf_func,
            gradient_func=gradient_func)

        # Set up sampler
        sampler = cuqi.experimental.mcmc.ULA(target, scale=1/dim**2)

        # Sample
        sampler.sample(2000)

    A Deblur example can be found in demos/demo27_ULA.py
    # TODO: update demo once sampler merged
    """

    _STATE_KEYS = Sampler._STATE_KEYS.union({'scale', 'current_target_grad'})

    def __init__(self, target=None, scale=1.0, **kwargs):

        super().__init__(target, **kwargs)
        self.initial_scale = scale

    def _initialize(self):
        self.scale = self.initial_scale
        self.current_target_grad = self._eval_target_grad(self.current_point)

    def validate_target(self):
        try:
            self._eval_target_grad(np.ones(self.dim))
            pass
        except (NotImplementedError, AttributeError):
            raise ValueError("The target needs to have a gradient method")

    def _eval_target_logd(self, x):
        return None

    def _eval_target_grad(self, x):
        return self.target.gradient(x)

    def _accept_or_reject(self, x_star, target_eval_star, target_grad_star):
        """
        Accepts the proposed state and updates the sampler's state accordingly, i.e.,
        current_point, current_target_eval, and current_target_grad_eval.

        Parameters
        ----------
        x_star : 
            The proposed state

        target_eval_star: 
            The log likelihood evaluated at x_star

        target_grad_star: 
            The gradient of log likelihood evaluated at x_star

        Returns
        -------
        scalar
            1 (accepted)
        """

        self.current_point = x_star
        self.current_target_grad = target_grad_star
        acc = 1

        return acc

    def step(self):
        # propose state
        xi = cuqi.distribution.Normal(mean=np.zeros(self.dim), std=np.sqrt(self.scale)).sample()
        x_star = self.current_point + 0.5*self.scale*self.current_target_grad + xi

        # evaluate target
        target_eval_star = self._eval_target_logd(x_star)
        target_grad_star = self._eval_target_grad(x_star)

        # accept or reject proposal
        acc = self._accept_or_reject(x_star, target_eval_star, target_grad_star)

        return acc

    def tune(self, skip_len, update_count):
        pass


class MALA(ULA): # Refactor to Proposal-based sampler?
    """  Metropolis-adjusted Langevin algorithm (MALA) (Roberts and Tweedie, 1996)

    Samples a distribution given its logd and gradient (up to a constant) based on
    Langevin diffusion dL_t = dW_t + 1/2*Nabla target.logd(L_t)dt,
    W_t is the `dim`-dimensional standard Brownian motion.
    A sample is firstly proposed by ULA and is then accepted or rejected according
    to a Metropolis–Hastings step.
    This accept-reject step allows us to remove the asymptotic bias of ULA.

    For more details see: Roberts, G. O., & Tweedie, R. L. (1996). Exponential convergence
    of Langevin distributions and their discrete approximations. Bernoulli, 341-363.

    Parameters
    ----------

    target : `cuqi.distribution.Distribution`
        The target distribution to sample. Must have logpdf and gradient method. Custom logpdfs 
        and gradients are supported by using a :class:`cuqi.distribution.UserDefinedDistribution`.
    
    initial_point : ndarray
        Initial parameters. *Optional*

    scale : float
        The Langevin diffusion discretization time step (In practice, scale must
        be smaller than 1/L, where L is the Lipschitz of the gradient of the log
        target density, logd).

    callback : callable, optional
        A function that will be called after each sampling step. It can be useful for monitoring the sampler during sampling.
        The function should take three arguments: the sampler object, the index of the current sampling step, the total number of requested samples. The last two arguments are integers. An example of the callback function signature is: `callback(sampler, sample_index, num_of_samples)`.


    Example
    -------
    .. code-block:: python

        # Parameters
        dim = 5 # Dimension of distribution
        mu = np.arange(dim) # Mean of Gaussian
        std = 1 # standard deviation of Gaussian

        # Logpdf function
        logpdf_func = lambda x: -1/(std**2)*np.sum((x-mu)**2)
        gradient_func = lambda x: -2/(std**2)*(x-mu)

        # Define distribution from logpdf as UserDefinedDistribution (sample and gradients also supported)
        target = cuqi.distribution.UserDefinedDistribution(dim=dim, logpdf_func=logpdf_func,
            gradient_func=gradient_func)

        # Set up sampler
        sampler = cuqi.experimental.mcmc.MALA(target, scale=1/5**2)

        # Sample
        sampler.sample(2000)

    A Deblur example can be found in demos/demo28_MALA.py
    # TODO: update demo once sampler merged
    """

    _STATE_KEYS = ULA._STATE_KEYS.union({'current_target_logd'})

    def _initialize(self):
        super()._initialize()
        self.current_target_logd = self.target.logd(self.current_point)

    def _eval_target_logd(self, x):
        return self.target.logd(x)

    def _accept_or_reject(self, x_star, target_eval_star, target_grad_star):
        """
        Accepts the proposed state according to a Metropolis step and updates
        the sampler's state accordingly, i.e., current_point, current_target_eval,
        and current_target_grad_eval.

        Parameters
        ----------
        x_star : 
            The proposed state

        target_eval_star: 
            The log likelihood evaluated at x_star

        target_grad_star: 
            The gradient of log likelihood evaluated at x_star

        Returns
        -------
        scaler
            1 if accepted, 0 otherwise
        """
        log_target_ratio = target_eval_star - self.current_target_logd
        log_prop_ratio = self._log_proposal(self.current_point, x_star, target_grad_star) \
            - self._log_proposal(x_star, self.current_point,  self.current_target_grad)
        log_alpha = min(0, log_target_ratio + log_prop_ratio)

        # accept/reject with Metropolis
        acc = 0
        log_u = np.log(np.random.rand())
        if (log_u <= log_alpha) and \
           (not np.isnan(target_eval_star)) and \
           (not np.isinf(target_eval_star)):
            self.current_point = x_star
            self.current_target_logd = target_eval_star
            self.current_target_grad = target_grad_star
            acc = 1
        return acc

    def tune(self, skip_len, update_count):
        pass

    def _log_proposal(self, theta_star, theta_k, g_logpi_k):
        mu = theta_k + ((self.scale)/2)*g_logpi_k
        misfit = theta_star - mu
        return -0.5*((1/(self.scale))*(misfit.T @ misfit))


class MYULA(ULA):
    """Moreau-Yoshida Unadjusted Langevin algorithm (MYUULA) (Durmus et al., 2018)

    Samples a smoothed target distribution given its smoothed logpdf gradient.
    It is based on the Langevin diffusion dL_t = dW_t + 1/2*Nabla target.logd(L_t)dt, 
    where W_t is a `dim`-dimensional standard Brownian motion.
    It targets a differentiable density (partially) smoothed by the Moreau-Yoshida
    envelope. The smoothed target density can be made arbitrarily closed to the
    true unsmoothed target density.

    For more details see: Durmus, Alain, Eric Moulines, and Marcelo Pereyra.
    "Efficient Bayesian
    computation by proximal Markov chain Monte Carlo: when Langevin meets Moreau."
    SIAM Journal on Imaging Sciences 11.1 (2018): 473-506.

    Parameters
    ----------

    target : `cuqi.distribution.Distribution`
        The target distribution to sample from. The target distribution results from
        a differentiable likelihood and prior of type RestorationPrior.
    
    initial_point : ndarray
        Initial parameters. *Optional*

    scale : float
        The Langevin diffusion discretization time step (In practice, scale must
        be smaller than 1/L, where L is the Lipschitz of the gradient of the log
        target density, logd).
        
    smoothing_strength : float
        This parameter controls the smoothing strength of MYULA.

    callback : callable, optional
        A function that will be called after each sampling step. It can be useful for monitoring the sampler during sampling.
        The function should take three arguments: the sampler object, the index of the current sampling step, the total number of requested samples. The last two arguments are integers. An example of the callback function signature is: `callback(sampler, sample_index, num_of_samples)`.

    A Deblur example can be found in demos/howtos/myula.py
    # TODO: update demo once sampler merged
    """
    def __init__(self, target=None, scale=1.0, smoothing_strength=0.1, **kwargs):
        self.smoothing_strength = smoothing_strength
        super().__init__(target=target, scale=scale, **kwargs)

    @Sampler.target.setter
    def target(self, value):
        """ Set the target density. Runs validation of the target. """
        self._target = value

        if self._target is not None:
            # Create a smoothed target
            self._smoothed_target = self._create_smoothed_target(value)

            # Validate the target
            self.validate_target()

    def _create_smoothed_target(self, value):
        """ Create a smoothed target using a Moreau-Yoshida envelope. """
        copied_value = copy(value)
        if isinstance(copied_value.prior, RestorationPrior):
            # Acceess the prior name
            name = value.prior.name
            copied_value.prior = MoreauYoshidaPrior(
                copied_value.prior,
                self.smoothing_strength,
                name=name)
        return copied_value

    def validate_target(self):
        # Call ULA target validation
        super().validate_target()

        # Additional validation for MYULA target
        if isinstance(self.target.prior, MoreauYoshidaPrior):
            raise ValueError(("The prior is already smoothed, apply"
                              " ULA when using a MoreauYoshidaPrior."))
        if not hasattr(self.target.prior, "restore"):
            raise NotImplementedError(
                ("Using MYULA with a prior that does not have a restore method"
                " is not supported.")
            )

    def _eval_target_grad(self, x):
        return self._smoothed_target.gradient(x)

class PnPULA(MYULA):
    """Plug-and-Play Unadjusted Langevin algorithm (PnP-ULA)
    (Laumont et al., 2022)

    Samples a smoothed target distribution given its smoothed logpdf gradient based on
    Langevin diffusion dL_t = dW_t + 1/2*Nabla target.logd(L_t)dt, where W_t is
    a `dim`-dimensional standard Brownian motion.
    It targets a differentiable density (partially) smoothed by a convolution
    with Gaussian kernel with zero mean and smoothing_strength variance. The
    smoothed target density can be made arbitrarily closed to the
    true unsmoothed target density. 

    For more details see: Laumont, R., Bortoli, V. D., Almansa, A., Delon, J.,
    Durmus, A., & Pereyra, M. (2022). Bayesian imaging using plug & play priors:
    when Langevin meets Tweedie. SIAM Journal on Imaging Sciences, 15(2), 701-737.

    Parameters
    ----------

    target : `cuqi.distribution.Distribution`
        The target distribution to sample. The target distribution result from
        a differentiable likelihood and prior of type RestorationPrior.
    
    initial_point : ndarray
        Initial parameters. *Optional*

    scale : float
        The Langevin diffusion discretization time step (In practice, a scale of
        1/L, where L is the Lipschitz of the gradient of the log target density
        is recommended but not guaranteed to be the optimal choice).
        
    smoothing_strength : float
        This parameter controls the smoothing strength of PnP-ULA.


    callback : callable, optional
        A function that will be called after each sampling step. It can be useful for monitoring the sampler during sampling.
        The function should take three arguments: the sampler object, the index of the current sampling step, the total number of requested samples. The last two arguments are integers. An example of the callback function signature is: `callback(sampler, sample_index, num_of_samples)`.

    # TODO: update demo once sampler merged
    """
    def __init__ (self, target=None, scale=1.0, smoothing_strength=0.1, **kwargs):
        super().__init__(target=target, scale=scale, 
                         smoothing_strength=smoothing_strength, **kwargs)
