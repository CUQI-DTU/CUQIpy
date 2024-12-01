import cuqi
import inspect

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

    return valid_samplers