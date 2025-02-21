import cuqi
import inspect
import numpy as np

# This import makes suggest_sampler easier to read
import cuqi.experimental.mcmc as samplers 

def find_valid_samplers(target, as_string = True):
        """
        Finds all possible sampleras that can be used for sampling from the target distribution.

        Parameters
        ----------

        target : `cuqi.distribution.Distribution`
            The target distribution to sample.
        
        as_string : boolean
            Whether to return the name of the sampler as a string instead of instantiating a sampler. *Optional*

    """

    all_samplers = [(name, cls) for name, cls in inspect.getmembers(cuqi.experimental.mcmc, inspect.isclass) if issubclass(cls, cuqi.experimental.mcmc.Sampler)]
    valid_samplers = []

    for name, sampler in all_samplers:
        try:
            sampler(target)
            valid_samplers += [name if as_string else sampler]
        except:
            pass

    # Need a separate case for HybridGibbs
    if find_valid_sampling_strategy(target) is not None:
        valid_samplers += [cuqi.experimental.mcmc.HybridGibbs.__name__ if as_string else cuqi.experimental.mcmc.HybridGibbs]

    return valid_samplers

def find_valid_sampling_strategy(target, as_string = True):
    """
        Find all possible sampling strategies to be used with the HybridGibbs sampler.
        Returns None if no sampler could be suggested for at least one conditional distribution.

        Parameters
        ----------

        target : `cuqi.distribution.JointDistribution`
            The target distribution to find a valid sampler strategy for.
        
        as_string : boolean
            Whether to return the name of the samplers in the sampling strategy as a string instead of instantiating samplers. *Optional*
    

    """

    if not isinstance(target, cuqi.distribution.JointDistribution):
        return None

    par_names = target.get_parameter_names()

    valid_samplers = dict()
    for par_name in par_names:
        conditional_params = {par_name_: np.ones(target.dim[i]) for i, par_name_ in enumerate(par_names) if par_name_ != par_name}
        conditional = target(**conditional_params)

        samplers = find_valid_samplers(conditional, as_string)
        if len(samplers) == 0:
            return None
        
        valid_samplers[par_name] = samplers

    return valid_samplers

def suggest_sampler(target, as_string = False, exceptions = []):
    """
        Suggests a possible sampler that can be used for sampling from the target distribution.
        Return None if no sampler could be suggested.

        Parameters
        ----------

        target : `cuqi.distribution.Distribution`
            The target distribution to sample.
        
        as_string : boolean
            Whether to return the name of the sampler as a string instead of instantiating a sampler. *Optional*
    
        exceptions : list of cuqi.experimental.mcmc sampler classes
            Samplers not to consider for suggestion. *Optional*

    """

    # TODO: Conditions for suggestions, e.g., only use CWMH when the dimension of the problem is low
    # TODO: Discuss ordering

    # Samplers with suggested default values (when no defaults are defined)
    ordering = [
        # Direct and Conjugate samplers
        (samplers.Direct, {}),
        (samplers.Conjugate, {}),
        (samplers.ConjugateApprox, {}),
        # Specialized samplers
        (samplers.LinearRTO, {}),
        (samplers.RegularizedLinearRTO, {}),
        (samplers.UGLA, {}),
        # Gradient.based samplers (Hamiltonian and Langevin)
        (samplers.NUTS, {}),
        (samplers.MALA, {}),
        (samplers.ULA, {}),
        # Gibbs and Componentwise samplers
        (samplers.HybridGibbs, {"sampling_strategy" : suggest_sampling_strategy(target, as_string = False)}),
        (samplers.CWMH, {"scale" : 0.05*np.ones(target.dim),
                         "x0" : 0.5*np.ones(target.dim)}),
        # Proposal based samplers
        (samplers.PCN, {"scale" : 0.02}),
        (samplers.MH, {}),
    ]

    valid_samplers = find_valid_samplers(target, as_string = False)
    
    for suggestion, values in ordering:
        if suggestion in valid_samplers and suggestion not in exceptions:
            # Sampler found
            if as_string:
                return suggestion.__name__
            else:
                return suggestion(target, **values)

    # No sampler can be suggested
    return None

def suggest_sampling_strategy(target, as_string = False):
    """
        Suggests a possible sampling strategy to be used with the HybridGibbs sampler.
        Returns None if no sampler could be suggested for at least one conditional distribution.

        Parameters
        ----------

        target : `cuqi.distribution.JointDistribution`
            The target distribution get a sampling strategy for.
        
        as_string : boolean
            Whether to return the name of the samplers in the sampling strategy as a string instead of instantiating samplers. *Optional*
    
    """

    if not isinstance(target, cuqi.distribution.JointDistribution):
        return None

    par_names = target.get_parameter_names()

    suggested_samplers = dict()
    for par_name in par_names:
        conditional_params = {par_name_: np.ones(target.dim[i]) for i, par_name_ in enumerate(par_names) if par_name_ != par_name}
        conditional = target(**conditional_params)

        sampler = suggest_sampler(conditional, as_string = as_string)
        if sampler is None:
            return None
        
        suggested_samplers[par_name] = sampler

    return suggested_samplers