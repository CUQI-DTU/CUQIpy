import cuqi
import inspect
import numpy as np

def find_valid_samplers(target):
    """ Finds all samplers in the cuqi.experimental.mcmc module that accept the provided target. """

    all_samplers = [(name, cls) for name, cls in inspect.getmembers(cuqi.experimental.mcmc, inspect.isclass) if issubclass(cls, cuqi.experimental.mcmc.Sampler)]
    valid_samplers = []

    for name, sampler in all_samplers:
        try:
            sampler(target)
            valid_samplers += [name]
        except:
            pass

    # Need a separate case for HybridGibbs
    if find_valid_sampling_strategy(target) is not None:
        valid_samplers += [cuqi.experimental.mcmc.HybridGibbs.__name__]

    return valid_samplers

def find_valid_sampling_strategy(target):
    """
        Find valid samplers to be used for creating a sampling strategy for the HybridGibbs sampler
    """

    if not isinstance(target, cuqi.distribution.JointDistribution):
        return None

    par_names = target.get_parameter_names()
    print(par_names)

    valid_samplers = dict()
    for par_name in par_names:
        conditional_params = {par_name_: np.ones(target.dim[i]) for i, par_name_ in enumerate(par_names) if par_name_ != par_name}
        conditional = target(**conditional_params)

        samplers = find_valid_samplers(conditional)
        if len(samplers) == 0:
            return None
        
        valid_samplers[par_name] = samplers

    return valid_samplers