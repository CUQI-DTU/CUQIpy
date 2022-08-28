from cuqi.distribution import JointDistribution
from cuqi.sampler import Sampler
from cuqi.samples import Samples
from typing import Dict, Union
import numpy as np
import sys


class Gibbs:

    def __init__(self, target: JointDistribution, sampling_strategy: Dict[Union[str,tuple], Sampler]):

        # Store target and allow conditioning to reduce to a single density
        self.target = target
        self.target._allow_reduce = True

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

        # Allocate memory for samples
        self.samples_warmup = self._allocate_samples(Nb)
        self.samples = self._allocate_samples(Ns)

        # Initial points
        current_samples = self._get_initial_points()

        # Sample tuning phase
        for i in range(Nb):
            current_samples = self.step_tune(current_samples)
            self._store_samples(self.samples_warmup, current_samples, i)
            self._print_progress(i+1, Nb, 'Warmup')

        # Sample phase
        for i in range(Ns):
            current_samples = self.step(current_samples)
            self._store_samples(self.samples, current_samples, i)
            self._print_progress(i+1, Ns, 'Sample')

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
        samples = {}
        for par_name in self.par_names:
            samples[par_name] = np.zeros((self.target.get_density(par_name).dim, Ns))
        return samples

    def _get_initial_points(self):
        """ Get initial points for each parameter """
        initial_points = {}
        for par_name in self.par_names:
            if hasattr(self.target.get_density(par_name), 'init_point'):
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
