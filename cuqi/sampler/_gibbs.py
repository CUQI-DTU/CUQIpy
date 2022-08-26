from cuqi.distribution import JointDistribution
from cuqi.sampler import Sampler
from cuqi.samples import Samples
from typing import Dict, Union
import numpy as np
import sys


class Gibbs:

    def __init__(self, target: JointDistribution, samplers: Dict[Union[str,tuple], Sampler]):

        # Store target and allow conditioning to reduce to a single density
        self.target = target
        self.target._allow_reduce = True

        # Parse samplers and split any keys that are tuple into separate keys
        self.samplers = {}
        for par_name in samplers.keys():
            if isinstance(par_name, tuple):
                for par_name_ in par_name:
                    self.samplers[par_name_] = samplers[par_name]
            else:
                self.samplers[par_name] = samplers[par_name]

    def sample(self, Ns, Nb=0):

        # First extract the names of the parameters we are sampling
        par_names = self.target.get_parameter_names()

        # Initialize samples
        samples = {}
        for par_name in par_names:
            samples[par_name] = np.zeros((self.target.get_density(par_name).dim, Ns+Nb))

        # Initial points
        for par_name in par_names:
            if hasattr(self.target.get_density(par_name), 'init_point'):
                samples[par_name][:,0] = self.target.get_density(par_name).init_point
            else:
                samples[par_name][:,0] = np.ones(self.target.get_density(par_name).dim)

        # Create dictionary of current samples
        current_samples = {}
        for par_name in par_names:
            current_samples[par_name] = samples[par_name][:,0]

        # Sample
        for i in range(Ns+Nb-1):

            # Sample from each conditional distribution
            for par_name in par_names:
                # dict of all other parameters to condition on
                other_params = {par_name_: current_samples[par_name_] for par_name_ in par_names if par_name_ != par_name}

                # Set up sampler for current conditional distribution
                sampler = self.samplers[par_name](self.target(**other_params))

                # Take a MCMC step
                current_samples[par_name] = sampler.step(current_samples[par_name])

                # Ensure even 1-dimensional samples are 1D arrays
                current_samples[par_name] = current_samples[par_name].reshape(-1)

            # Store current samples
            for par_name in par_names:
                samples[par_name][:, i+1] = current_samples[par_name]

            self._print_progress(i+2,Ns+Nb)

        # Convert samples to cuqi.samples.Samples object with correct geometry
        for par_name in par_names:
            samples[par_name] = Samples(samples[par_name], self.target.get_density(par_name).geometry)
            samples[par_name] = samples[par_name].burnthin(Nb)

        return samples


    def _print_progress(self,s,Ns):
        """Prints sampling progress"""
        if Ns > 2:
            if (s % (max(Ns//100,1))) == 0:
                msg = f'Sample {s} / {Ns}'
                sys.stdout.write('\r'+msg)
            if s==Ns:
                msg = f'Sample {s} / {Ns}'
                sys.stdout.write('\r'+msg+'\n')
