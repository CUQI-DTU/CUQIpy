from cuqi.distribution import JointDistribution
from cuqi.experimental.mcmc import Sampler
from cuqi.samples import Samples
from cuqi.experimental.mcmc import NUTS
from typing import Dict
import numpy as np
import warnings

try:
    from progressbar import progressbar
except ImportError:
    def progressbar(iterable, **kwargs):
        warnings.warn("Module mcmc: Progressbar not found. Install progressbar2 to get sampling progress.")
        return iterable

# Not subclassed from Sampler as Gibbs handles multiple samplers and samples multiple parameters
# Similar approach as for JointDistribution
class HybridGibbs: 
    """
    Hybrid Gibbs sampler for sampling a joint distribution.

    Gibbs sampling samples the variables of the distribution sequentially,
    one variable at a time. When a variable represents a random vector, the
    whole vector is sampled simultaneously.
    
    The sampling of each variable is done by sampling from the conditional
    distribution of that variable given the values of the other variables.
    This is often a very efficient way of sampling from a joint distribution
    if the conditional distributions are easy to sample from. 

    Hybrid Gibbs sampler is a generalization of the Gibbs sampler where the
    conditional distributions are sampled using different MCMC samplers.
    
    When the conditionals are sampled exactly, the samples from the Gibbs 
    sampler converge to the joint distribution. See e.g.
    Gelman et al. "Bayesian Data Analysis" (2014), Third Edition
    for more details.

    In each Gibbs step, the corresponding sampler has the initial_point 
    and initial_scale (if applicable) set to the value of the previous step
    and the sampler is reinitialized. This means that the sampling is not 
    fully stateful at this point. This means samplers like NUTS will lose
    their internal state between Gibbs steps.

    Parameters
    ----------
    target : cuqi.distribution.JointDistribution
        Target distribution to sample from.
    
    sampling_strategy : dict
        Dictionary of sampling strategies for each variable.
        Keys are variable names.
        Values are sampler objects.

    num_sampling_steps : dict, *optional*
        Dictionary of number of sampling steps for each variable.
        The sampling steps are defined as the number of times the sampler
        will call its step method in each Gibbs step.
        Default is 1 for all variables.

    Example
    -------
    .. code-block:: python

        import cuqi
        import numpy as np

        # Model and data
        A, y_obs, probinfo = cuqi.testproblem.Deconvolution1D(phantom='square').get_components()
        n = A.domain_dim

        # Define distributions
        d = cuqi.distribution.Gamma(1, 1e-4)
        l = cuqi.distribution.Gamma(1, 1e-4)
        x = cuqi.distribution.GMRF(np.zeros(n), lambda d: d)
        y = cuqi.distribution.Gaussian(A, lambda l: 1/l)

        # Combine into a joint distribution and create posterior
        joint = cuqi.distribution.JointDistribution(d, l, x, y)
        posterior = joint(y=y_obs)

        # Define sampling strategy
        sampling_strategy = {
            'x': cuqi.experimental.mcmc.LinearRTO(maxit=15),
            'd': cuqi.experimental.mcmc.Conjugate(),
            'l': cuqi.experimental.mcmc.Conjugate(),
        }

        # Define Gibbs sampler
        sampler = cuqi.experimental.mcmc.HybridGibbs(posterior, sampling_strategy)

        # Run sampler
        samples = sampler.sample(Ns=1000, Nb=200)

        # Plot results
        samples['x'].plot_ci(exact=probinfo.exactSolution)
        samples['d'].plot_trace(figsize=(8,2))
        samples['l'].plot_trace(figsize=(8,2))
            
    """

    def __init__(self, target: JointDistribution, sampling_strategy: Dict[str, Sampler], num_sampling_steps: Dict[str, int] = None):

        # Store target and allow conditioning to reduce to a single density
        self.target = target() # Create a copy of target distribution (to avoid modifying the original)

        # Store sampler instances (again as a copy to avoid modifying the original)
        self.samplers = sampling_strategy.copy()

        # Store number of sampling steps for each parameter
        self.num_sampling_steps = num_sampling_steps

        # Store parameter names
        self.par_names = self.target.get_parameter_names()

        # Initialize sampler (after target is set)
        self._initialize()

    def _initialize(self):
        """ Initialize sampler """

        # Initial points
        self.current_samples = self._get_initial_points()

        # Initialize sampling steps
        self._initialize_num_sampling_steps()

        # Allocate samples
        self._allocate_samples()

        # Set targets
        self._set_targets()

        # Initialize the samplers
        self._initialize_samplers() 

        # Run over pre-sample methods for samplers that have it
        # TODO. Some samplers (NUTS) seem to require to run _pre_warmup before _pre_sample
        # This is not ideal and should be fixed in the future
        for sampler in self.samplers.values():
            self._pre_warmup_and_pre_sample_sampler(sampler)

        # Validate all targets for samplers.
        self.validate_targets()

    # ------------ Public methods ------------
    def validate_targets(self):
        """ Validate each of the conditional targets used in the Gibbs steps """
        if not isinstance(self.target, JointDistribution):
            raise ValueError('Target distribution must be a JointDistribution.')
        for sampler in self.samplers.values():
            sampler.validate_target()

    def sample(self, Ns) -> 'HybridGibbs':
        """ Sample from the joint distribution using Gibbs sampling """
        for _ in progressbar(range(Ns)):
            self.step()
            self._store_samples()

    def warmup(self, Nb, tune_freq=0.1) -> 'HybridGibbs':

        tune_interval = max(int(tune_freq * Nb), 1)

        """ Warmup (tune) the Gibbs sampler """
        for idx in progressbar(range(Nb)):
            self.step()
            # Tune the sampler at tuning intervals
            if (idx + 1) % tune_interval == 0:
                self.tune(tune_interval, idx // tune_interval) 
            self._store_samples()

    def get_samples(self) -> Dict[str, Samples]:
        samples_object = {}
        for par_name in self.par_names:
            samples_array = np.array(self.samples[par_name]).T
            samples_object[par_name] = Samples(samples_array, self.target.get_density(par_name).geometry)
        return samples_object
    
    def step(self):
        """ Sequentially go through all parameters and sample them conditionally on each other """

        # Sample from each conditional distribution
        for par_name in self.par_names:

            # Set target for current parameter
            self._set_target(par_name)

            # Get sampler
            sampler = self.samplers[par_name]

            # Instead of simply changing the target of the sampler, we reinitialize it.
            # This is to ensure that all internal variables are set to match the new target.
            # To return the sampler to the old state and history, we first extract the state and history
            # before reinitializing the sampler and then set the state and history back to the sampler

            # Extract state and history from sampler
            if isinstance(sampler, NUTS): # Special case for NUTS as it is not playing nice with get_state and get_history
                sampler.initial_point = sampler.current_point
            else:
                sampler_state = sampler.get_state()
                sampler_history = sampler.get_history()

            # Reinitialize sampler
            sampler.reinitialize()

            # Set state and history back to sampler
            if not isinstance(sampler, NUTS): # Again, special case for NUTS.
                sampler.set_state(sampler_state)
                sampler.set_history(sampler_history)

            # Run pre_warmup and pre_sample methods for sampler
            # TODO. Some samplers (NUTS) seem to require to run _pre_warmup before _pre_sample
            self._pre_warmup_and_pre_sample_sampler(sampler)

            # Take MCMC steps
            for _ in range(self.num_sampling_steps[par_name]):
                acc = sampler.step()
                self.samplers[par_name]._acc.append(acc)

            # Extract samples (Ensure even 1-dimensional samples are 1D arrays)
            self.current_samples[par_name] = sampler.current_point.reshape(-1)

    def tune(self, skip_len, idx):
        """ Tune each of the samplers """
        for par_name in self.par_names:
            self.samplers[par_name].tune(skip_len=skip_len, update_count=idx) # When hardcoded like this, the acceptance rate is always 1 or 0, giving issues in the tuning method

    # ------------ Private methods ------------
    def _initialize_samplers(self):
        """ Initialize samplers """
        for sampler in self.samplers.values():
            sampler.initialize()

    def _initialize_num_sampling_steps(self):
        """ Initialize the number of sampling steps for each sampler. Defaults to 1 if not set by user """

        if self.num_sampling_steps is None:
            self.num_sampling_steps = {par_name: 1 for par_name in self.par_names}

        for par_name in self.par_names:
            if par_name not in self.num_sampling_steps:
                self.num_sampling_steps[par_name] = 1


    def _pre_warmup_and_pre_sample_sampler(self, sampler):
        if hasattr(sampler, '_pre_warmup'): sampler._pre_warmup()
        if hasattr(sampler, '_pre_sample'): sampler._pre_sample()

    def _set_targets(self):
        """ Set targets for all samplers using the current samples """
        par_names = self.par_names
        for par_name in par_names:
            self._set_target(par_name)

    def _set_target(self, par_name):
        """ Set target conditional distribution for a single parameter using the current samples """
        # Get all other conditional parameters other than the current parameter and update the target
        # This defines - from a joint p(x,y,z) - the conditional distribution p(x|y,z) or p(y|x,z) or p(z|x,y)
        conditional_params = {par_name_: self.current_samples[par_name_] for par_name_ in self.par_names if par_name_ != par_name}
        self.samplers[par_name].target = self.target(**conditional_params)

    def _allocate_samples(self):
        """ Allocate memory for samples """
        samples = {}
        for par_name in self.par_names:
            samples[par_name] = []
        self.samples = samples

    def _get_initial_points(self):
        """ Get initial points for each parameter """
        initial_points = {}
        for par_name in self.par_names:
            sampler = self.samplers[par_name]
            if sampler.initial_point is None:
                sampler.initial_point = sampler._get_default_initial_point(self.target.get_density(par_name).dim)
            initial_points[par_name] = sampler.initial_point
            
        return initial_points

    def _store_samples(self):
        """ Store current samples at index i of samples dict """
        for par_name in self.par_names:
            self.samples[par_name].append(self.current_samples[par_name])
