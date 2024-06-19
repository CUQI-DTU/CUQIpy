import numpy as np
import cuqi
from cuqi.experimental.mcmc import SamplerNew
from cuqi.array import CUQIarray

class ULANew(SamplerNew): # Refactor to Proposal-based sampler?
    """Unadjusted Langevin algorithm (ULA) (Roberts and Tweedie, 1996)

    Samples a distribution given its logpdf and gradient (up to a constant) based on
    Langevin diffusion dL_t = dW_t + 1/2*Nabla target.logd(L_t)dt,  where L_t is 
    the Langevin diffusion and W_t is the `dim`-dimensional standard Brownian motion.

    For more details see: Roberts, G. O., & Tweedie, R. L. (1996). Exponential convergence
    of Langevin distributions and their discrete approximations. Bernoulli, 341-363.

    Parameters
    ----------

    target : `cuqi.distribution.Distribution`
        The target distribution to sample. Must have logd and gradient method. Custom logpdfs 
        and gradients are supported by using a :class:`cuqi.distribution.UserDefinedDistribution`.
    
    initial_point : ndarray
        Initial parameters. *Optional*

    scale : int
        The Langevin diffusion discretization time step (In practice, a scale of 1/dim**2 is
        recommended but not guaranteed to be the optimal choice).

    callback : callable, *Optional*
        If set this function will be called after every sample.
        The signature of the callback function is `callback(sample, sample_index)`,
        where `sample` is the current sample and `sample_index` is the index of the sample.
        An example is shown in demos/demo31_callback.py.


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
        sampler = cuqi.experimental.mcmc.ULANew(target, scale=1/dim**2)

        # Sample
        sampler.sample(2000)

    A Deblur example can be found in demos/demo27_ULA.py
    # TODO: update demo once sampler merged
    """

    _STATE_KEYS = SamplerNew._STATE_KEYS.union({'current_target_logd', 'scale', 'current_target_grad'})

    def __init__(self, target=None, scale=1.0, **kwargs):

        super().__init__(target, **kwargs)

        self.initial_scale = scale

    def _initialize(self):
        self.scale = self.initial_scale
        self.current_target_logd = self.target.logd(self.current_point)
        self.current_target_grad = self.target.gradient(self.current_point)

    def validate_target(self):
        try:
            self.target.gradient(np.ones(self.dim))
            pass
        except (NotImplementedError, AttributeError):
            raise ValueError("The target needs to have a gradient method")

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
        self.current_target_logd = target_eval_star
        self.current_target_grad = target_grad_star
        acc = 1
        return acc

    def step(self):
        # propose state
        xi = cuqi.distribution.Normal(mean=np.zeros(self.dim), std=np.sqrt(self.scale)).sample()
        x_star = self.current_point + 0.5*self.scale*self.current_target_grad + xi

        # evaluate target
        target_eval_star, target_grad_star = self.target.logd(x_star), self.target.gradient(x_star)

        # accept or reject proposal
        acc = self._accept_or_reject(x_star, target_eval_star, target_grad_star)

        return acc

    def tune(self, skip_len, update_count):
        pass


class MALANew(ULANew): # Refactor to Proposal-based sampler?
    """  Metropolis-adjusted Langevin algorithm (MALA) (Roberts and Tweedie, 1996)

    Samples a distribution given its logd and gradient (up to a constant) based on
    Langevin diffusion dL_t = dW_t + 1/2*Nabla target.logd(L_t)dt,  where L_t is 
    the Langevin diffusion and W_t is the `dim`-dimensional standard Brownian motion. 
    The sample is then accepted or rejected according to Metropolis–Hastings algorithm.

    For more details see: Roberts, G. O., & Tweedie, R. L. (1996). Exponential convergence
    of Langevin distributions and their discrete approximations. Bernoulli, 341-363.

    Parameters
    ----------

    target : `cuqi.distribution.Distribution`
        The target distribution to sample. Must have logpdf and gradient method. Custom logpdfs 
        and gradients are supported by using a :class:`cuqi.distribution.UserDefinedDistribution`.
    
    initial_point : ndarray
        Initial parameters. *Optional*

    scale : int
        The Langevin diffusion discretization time step.

    callback : callable, *Optional*
        If set this function will be called after every sample.
        The signature of the callback function is `callback(sample, sample_index)`,
        where `sample` is the current sample and `sample_index` is the index of the sample.
        An example is shown in demos/demo31_callback.py.


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
        sampler = cuqi.experimental.mcmc.MALANew(target, scale=1/5**2)

        # Sample
        sampler.sample(2000)

    A Deblur example can be found in demos/demo28_MALA.py
    # TODO: update demo once sampler merged
    """

    def _accept_or_reject(self, x_star, target_eval_star, target_grad_star):
        """
        Accepts the proposed state according to a Metropolis step and updates the sampler's state accordingly, i.e., current_point, current_target_eval, and current_target_grad_eval.

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
        if (log_u <= log_alpha) and (np.isnan(target_eval_star) == False):
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
