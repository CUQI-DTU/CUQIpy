from cuqi.distribution import JointDistribution
from cuqi.sampler import Sampler
from cuqi.samples import Samples
from typing import Dict, Union
import numpy as np
import sys


class Gibbs:
    """
    Gibbs sampler for sampling a joint distribution.

    Gibbs sampling samples the variables of the distribution sequentially,
    one variable at a time. When a variable represents a random vector, the
    whole vector is sampled simultaneously.
    
    The sampling of each variable is done by sampling from the conditional
    distribution of that variable given the values of the other variables.
    This is often a very efficient way of sampling from a joint distribution
    if the conditional distributions are easy to sample from.

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
            'x': cuqi.sampler.LinearRTO,
            ('d', 'l'): cuqi.sampler.Conjugate,
        }

        # Define Gibbs sampler
        sampler = cuqi.sampler.Gibbs(posterior, sampling_strategy)

        # Run sampler
        samples = sampler.sample(Ns=1000, Nb=200)

        # Plot results
        samples['x'].plot_ci(exact=probinfo.exactSolution)
        samples['d'].plot_trace(figsize=(8,2))
        samples['l'].plot_trace(figsize=(8,2))
            
    """

    def __init__(self, target: JointDistribution, sampling_strategy: Dict[Union[str,tuple], Sampler]):

        # Store target and allow conditioning to reduce to a single density
        self.target = target() # Create a copy of target distribution (to avoid modifying the original)

        # Parse samplers and split any keys that are tuple into separate keys
        self.samplers = {}
        for par_name in sampling_strategy.keys():
            if isinstance(par_name, tuple):
                for par_name_ in par_name:
                    self.samplers[par_name_] = sampling_strategy[par_name]
            else:
                self.samplers[par_name] = sampling_strategy[par_name]

        # Store parameter names
        self.par_names = self.target.get_parameter_names()

    # ------------ Public methods ------------
    def sample(self, Ns, Nb=0):
        """ Sample from target distribution """

        # Initial points
        current_samples = self._get_initial_points()

        # Compute how many samples were already taken previously
        at_Nb = self._Nb
        at_Ns = self._Ns

        # Allocate memory for samples
        self._allocate_samples_warmup(Nb)
        self._allocate_samples(Ns)

        # Sample tuning phase
        for i in range(at_Nb, at_Nb+Nb):
            current_samples = self.step_tune(current_samples)
            self._store_samples(self.samples_warmup, current_samples, i)
            self._print_progress(i+1+at_Nb, at_Nb+Nb, 'Warmup')

        # Sample phase
        for i in range(at_Ns, at_Ns+Ns):
            current_samples = self.step(current_samples)
            self._store_samples(self.samples, current_samples, i)
            self._print_progress(i+1, at_Ns+Ns, 'Sample')

        # Convert to samples objects and return
        return self._convert_to_Samples(self.samples)

    def step(self, current_samples):
        """ Sequentially go through all parameters and sample them conditionally on each other """

        # Extract par names
        par_names = self.par_names

        # Sample from each conditional distribution
        for par_name in par_names:

            # Dict of all other parameters to condition on
            other_params = {par_name_: current_samples[par_name_] for par_name_ in par_names if par_name_ != par_name}

            # Set up sampler for current conditional distribution
            sampler = self.samplers[par_name](self.target(**other_params))

            # Take a MCMC step
            current_samples[par_name] = sampler.step(current_samples[par_name])

            # Ensure even 1-dimensional samples are 1D arrays
            current_samples[par_name] = current_samples[par_name].reshape(-1)
        
        return current_samples

    def step_tune(self, current_samples):
        """ Perform a single MCMC step for each parameter and tune the sampler """
        # Not implemented. No tuning happening here yet. Requires samplers to be able to be modified after initialization.
        return self.step(current_samples)

    # ------------ Private methods ------------
    def _allocate_samples(self, Ns):
        """ Allocate memory for samples """
        # Allocate memory for samples
        samples = {}
        for par_name in self.par_names:
            samples[par_name] = np.zeros((self.target.get_density(par_name).dim, Ns))
        
        # Store samples in self
        if hasattr(self, 'samples'):
            # Append to existing samples (This makes a copy)
            for par_name in self.par_names:
                samples[par_name] = np.hstack((self.samples[par_name], samples[par_name]))
        self.samples = samples

    def _allocate_samples_warmup(self, Nb):
        """ Allocate memory for samples """
        
        # If we already have warmup samples and more are requested raise error
        if hasattr(self, 'samples_warmup') and Nb != 0:
            raise ValueError('Sampler already has run warmup phase. Cannot run warmup phase again.')

        # Allocate memory for samples
        samples = {}
        for par_name in self.par_names:
            samples[par_name] = np.zeros((self.target.get_density(par_name).dim, Nb))
        self.samples_warmup = samples

    def _get_initial_points(self):
        """ Get initial points for each parameter """
        initial_points = {}
        for par_name in self.par_names:
            if hasattr(self, 'samples'):
                initial_points[par_name] = self.samples[par_name][:, -1]
            elif hasattr(self, 'samples_warmup'):
                initial_points[par_name] = self.samples_warmup[par_name][:, -1]
            elif hasattr(self.target.get_density(par_name), 'init_point'):
                initial_points[par_name] = self.target.get_density(par_name).init_point
            else:
                initial_points[par_name] = np.ones(self.target.get_density(par_name).dim)
        return initial_points

    def _store_samples(self, samples, current_samples, i):
        """ Store current samples at index i of samples dict """
        for par_name in self.par_names:
            samples[par_name][:, i] = current_samples[par_name]

    def _convert_to_Samples(self, samples):
        """ Convert each parameter in samples dict to cuqi.samples.Samples object with correct geometry """
        samples_object = {}
        for par_name in self.par_names:
            samples_object[par_name] = Samples(samples[par_name], self.target.get_density(par_name).geometry)
        return samples_object

    def _print_progress(self, s, Ns, phase):
        """Prints sampling progress"""
        if Ns < 2: # Don't print progress if only one sample
            return
        if (s % (max(Ns//100,1))) == 0:
            msg = f'{phase} {s} / {Ns}'
            sys.stdout.write('\r'+msg)
        if s==Ns:
            msg = f'{phase} {s} / {Ns}'
            sys.stdout.write('\r'+msg+'\n')

    # ------------ Private properties ------------
    @property
    def _Ns(self):
        """ Number of samples already taken """
        if hasattr(self, 'samples'):
            return self.samples[self.par_names[0]].shape[-1]
        else:
            return 0
    
    @property
    def _Nb(self):
        """ Number of samples already taken in warmup phase """
        if hasattr(self, 'samples_warmup'):
            return self.samples_warmup[self.par_names[0]].shape[-1]
        else:
            return 0
