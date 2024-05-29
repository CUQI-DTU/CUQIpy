from cuqi.distribution import JointDistribution
from cuqi.experimental.mcmc import SamplerNew
from cuqi.samples import Samples
from typing import Dict
import numpy as np
import warnings

try:
    from progressbar import progressbar
except ImportError:
    def progressbar(iterable, **kwargs):
        warnings.warn("Module mcmc: Progressbar not found. Install progressbar2 to get sampling progress.")
        return iterable

# Not subclassed from SamplerNew as Gibbs handles multiple samplers and samples multiple parameters
# Similar approach as for JointDistribution
class GibbsNew: 
    """
    Gibbs sampler for sampling a joint distribution.

    Gibbs sampling samples the variables of the distribution sequentially,
    one variable at a time. When a variable represents a random vector, the
    whole vector is sampled simultaneously.
    
    The sampling of each variable is done by sampling from the conditional
    distribution of that variable given the values of the other variables.
    This is often a very efficient way of sampling from a joint distribution
    if the conditional distributions are easy to sample from.

    In each Gibbs step, the corresponding sampler has the initial_point set
    to the previous sample and the sampler is reinitialized.

    Parameters
    ----------
    target : cuqi.distribution.JointDistribution
        Target distribution to sample from.
    
    sampling_strategy : dict
        Dictionary of sampling strategies for each parameter.
        Keys are parameter names.
        Values are sampler objects.

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
            'x': cuqi.experimental.mcmc.LinearRTONew(maxit=15),
            'd': cuqi.experimental.mcmc.ConjugateNew(),
            'l': cuqi.experimental.mcmc.ConjugateNew(),
        }

        # Define Gibbs sampler
        sampler = cuqi.experimental.mcmc.GibbsNew(posterior, sampling_strategy)

        # Run sampler
        samples = sampler.sample(Ns=1000, Nb=200)

        # Plot results
        samples['x'].plot_ci(exact=probinfo.exactSolution)
        samples['d'].plot_trace(figsize=(8,2))
        samples['l'].plot_trace(figsize=(8,2))
            
    """

    def __init__(self, target: JointDistribution, sampling_strategy: Dict[str, SamplerNew]):

        # Store target and allow conditioning to reduce to a single density
        self.target = target() # Create a copy of target distribution (to avoid modifying the original)

        # Store sampler instances (again as a copy to avoid modifying the original)
        self.samplers = sampling_strategy.copy()

        # Store parameter names
        self.par_names = self.target.get_parameter_names()

        # Initialize sampler (after target is set)
        self._initialize()

    def _initialize(self):
        """ Initialize sampler """

        # Initial points
        self.current_samples = self._get_initial_points()

        # Allocate samples
        self._allocate_samples()

        # Set targets
        self._set_targets()

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

    def sample(self, Ns) -> 'GibbsNew':
        """ Sample from the joint distribution using Gibbs sampling """
        for _ in progressbar(range(Ns)):
            self.step()
            self._store_samples()

    def warmup(self, Nb) -> 'GibbsNew':
        """ Warmup (tune) the Gibbs sampler """
        for idx in progressbar(range(Nb)):
            self.step()
            self.tune(idx)
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

            # Set targets (TODO: This is inefficient. Instead we should only update the target for the current parameter)
            self._set_targets()

            # Get sampler
            sampler = self.samplers[par_name]

            # Set initial point using current samples and reinitalize sampler
            # This makes the sampler loose all of its state
            # We need to design tests that allow samplers to change target
            # and not require reinitialization. This is needed to keep properties
            # like the internal state of NUTS for the next Gibbs step.
            sampler.initial_point = self.current_samples[par_name]
            sampler.reinitialize()

            # Run pre_warmup and pre_sample methods for sampler
            # TODO. Some samplers (NUTS) seem to require to run _pre_warmup before _pre_sample
            self._pre_warmup_and_pre_sample_sampler(sampler)

            # Take a MCMC step
            sampler.step()

            # Extract samples (Ensure even 1-dimensional samples are 1D arrays)
            self.current_samples[par_name] = sampler.current_point.reshape(-1)

    def tune(self, idx):
        """ Tune each of the samplers """

        for par_name in self.par_names:
            self.samplers[par_name].tune(skip_len=1, update_count=idx)

    # ------------ Private methods ------------
    def _pre_warmup_and_pre_sample_sampler(self, sampler):
        if hasattr(sampler, '_pre_warmup'): sampler._pre_warmup()
        if hasattr(sampler, '_pre_sample'): sampler._pre_sample()

    def _set_targets(self):
        """ Set targets for all samplers """
        par_names = self.par_names
        for par_name in par_names:
            other_params = {par_name_: self.current_samples[par_name_] for par_name_ in par_names if par_name_ != par_name}
            self.samplers[par_name].target = self.target(**other_params)

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
            if hasattr(self.target.get_density(par_name), 'init_point'):
                initial_points[par_name] = self.target.get_density(par_name).init_point
            else:
                initial_points[par_name] = np.ones(self.target.get_density(par_name).dim)
        return initial_points

    def _store_samples(self):
        """ Store current samples at index i of samples dict """
        for par_name in self.par_names:
            self.samples[par_name].append(self.current_samples[par_name])
