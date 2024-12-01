import numpy as np
import cuqi
from cuqi.sampler import Sampler

class ULA(Sampler):
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
    
    x0 : ndarray
        Initial parameters. *Optional*

    scale : int
        The Langevin diffusion discretization time step (In practice, a scale of 1/dim**2 is
        recommended but not guaranteed to be the optimal choice).

    dim : int
        Dimension of parameter space. Required if target logpdf and gradient are callable 
        functions. *Optional*.

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
        sampler = cuqi.sampler.ULA(target, scale=1/dim**2)

        # Sample
        samples = sampler.sample(2000)

    A Deblur example can be found in demos/demo27_ULA.py
    """
    def __init__(self, target, scale, x0=None, dim=None, rng=None, **kwargs):
        super().__init__(target, x0=x0, dim=dim, **kwargs)
        self.scale = scale
        self.rng = rng

    def _sample_adapt(self, N, Nb):
        return self._sample(N, Nb)

    def _sample(self, N, Nb):    
        # allocation
        Ns = Nb+N
        samples = np.empty((self.dim, Ns))
        target_eval = np.empty(Ns)
        g_target_eval = np.empty((self.dim, Ns))
        acc = np.zeros(Ns)

        # initial state
        samples[:, 0] = self.x0
        target_eval[0], g_target_eval[:,0] = self.target.logd(self.x0), self.target.gradient(self.x0)
        acc[0] = 1

        # ULA
        for s in range(Ns-1):
            samples[:, s+1], target_eval[s+1], g_target_eval[:,s+1], acc[s+1] = \
                self.single_update(samples[:, s], target_eval[s], g_target_eval[:,s])            
            self._print_progress(s+2,Ns) #s+2 is the sample number, s+1 is index assuming x0 is the first sample
            self._call_callback(samples[:, s+1], s+1)
    
        # apply burn-in 
        samples = samples[:, Nb:]
        target_eval = target_eval[Nb:]
        acc = acc[Nb:]
        return samples, target_eval, np.mean(acc)

    def single_update(self, x_t, target_eval_t, g_target_eval_t):
        # approximate Langevin diffusion
        xi = cuqi.distribution.Normal(mean=np.zeros(self.dim), std=np.sqrt(self.scale)).sample(rng=self.rng)
        x_star = x_t + 0.5*self.scale*g_target_eval_t + xi
        logpi_eval_star, g_logpi_star = self.target.logd(x_star), self.target.gradient(x_star)

        # msg
        if np.isnan(logpi_eval_star):
            raise NameError('NaN potential func. Consider using smaller scale parameter')

        return x_star, logpi_eval_star, g_logpi_star, 1 # sample always accepted without Metropolis correction


class MALA(ULA):
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
    
    x0 : ndarray
        Initial parameters. *Optional*

    scale : int
        The Langevin diffusion discretization time step.

    dim : int
        Dimension of parameter space. Required if target logpdf and gradient are callable 
        functions. *Optional*.

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
        sampler = cuqi.sampler.MALA(target, scale=1/5**2)

        # Sample
        samples = sampler.sample(2000)

    A Deblur example can be found in demos/demo28_MALA.py
    """
    def __init__(self, target, scale, x0=None, dim=None, rng=None, **kwargs):
        super().__init__(target, scale, x0=x0, dim=dim, rng=rng, **kwargs)

    def single_update(self, x_t, target_eval_t, g_target_eval_t):
        # approximate Langevin diffusion
        xi = cuqi.distribution.Normal(mean=np.zeros(self.dim), std=np.sqrt(self.scale)).sample(rng=self.rng)
        x_star = x_t + (self.scale/2)*g_target_eval_t + xi
        logpi_eval_star, g_logpi_star = self.target.logd(x_star), self.target.gradient(x_star)

        # Metropolis step
        log_target_ratio = logpi_eval_star - target_eval_t
        log_prop_ratio = self.log_proposal(x_t, x_star, g_logpi_star) \
            - self.log_proposal(x_star, x_t,  g_target_eval_t)
        log_alpha = min(0, log_target_ratio + log_prop_ratio)

        # accept/reject
        log_u = np.log(cuqi.distribution.Uniform(low=0, high=1).sample(rng=self.rng))
        if (log_u <= log_alpha) and (np.isnan(logpi_eval_star) == False):
            return x_star, logpi_eval_star, g_logpi_star, 1
        else:
            return x_t.copy(), target_eval_t, g_target_eval_t.copy(), 0

    def log_proposal(self, theta_star, theta_k, g_logpi_k):
        mu = theta_k + ((self.scale)/2)*g_logpi_k
        misfit = theta_star - mu
        return -0.5*((1/(self.scale))*(misfit.T @ misfit))

